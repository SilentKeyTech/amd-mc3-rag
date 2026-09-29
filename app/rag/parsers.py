"""Defensive corpus parsers.

Every parser takes a path and returns a list of Section(text, kind, image) or
raises; the walker catches per file so one bad file never costs the rest of the
corpus. Office formats are read with zipfile + ElementTree (stdlib only), PDFs
with pypdf (pure Python). Nothing here needs a native wheel.
"""

from __future__ import annotations

import csv
import io
import os
import re
import stat
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

TEXT_EXT = {
    ".txt", ".log", ".md", ".rst", ".py", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".conf", ".sh", ".js", ".ts", ".c", ".h", ".cpp", ".hpp", ".java", ".go",
    ".rs", ".sql", ".xml", ".html", ".htm", ".tex", ".properties", ".env",
}
TABLE_EXT = {".csv", ".tsv"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
PDF_EXT = {".pdf"}
DOCX_EXT = {".docx"}
XLSX_EXT = {".xlsx", ".xlsm"}
PPTX_EXT = {".pptx"}

OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # encrypted Office files are OLE containers
MAX_TEXT_BYTES = 8 * 1024 * 1024


class Unreadable(Exception):
    """The file exists but must not be used (encrypted, unknown, empty)."""


@dataclass
class Section:
    text: str
    kind: str = "text"          # text | table | image
    image: bytes | None = None  # raw image bytes awaiting OCR
    label: str = ""             # e.g. "page 2", "sheet Parts", "embedded image1.png"


@dataclass
class ParsedFile:
    rel: str
    ext: str
    sections: list[Section] = field(default_factory=list)


def _ns_strip(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig",):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1")


def _looks_binary(raw: bytes) -> bool:
    if not raw:
        return False
    sample = raw[:4096]
    if b"\x00" in sample:
        return True
    ctrl = sum(1 for b in sample if b < 9 or 13 < b < 32)
    return ctrl / len(sample) > 0.05


def read_bytes(path: Path) -> bytes:
    with open(path, "rb") as fh:  # raises PermissionError on a mode-000 file
        return fh.read()


# ---------------------------------------------------------------- text / csv

def parse_text(path: Path, raw: bytes) -> list[Section]:
    if _looks_binary(raw):
        raise Unreadable("binary content in a text-typed file")
    return [Section(_decode(raw[:MAX_TEXT_BYTES]))]


def parse_table(path: Path, raw: bytes) -> list[Section]:
    text = _decode(raw[:MAX_TEXT_BYTES])
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        delim = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|").delimiter
    except csv.Error:
        pass
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim) if any(c.strip() for c in r)]
    return [Section(rows_to_text(rows), kind="table")] if rows else []


def rows_to_text(rows: list[list[str]], title: str = "") -> str:
    """Render a table as one self-describing line per row: header=value; ..."""
    if not rows:
        return ""
    header = [h.strip() for h in rows[0]]
    has_header = len(rows) > 1 and all(h and not re.fullmatch(r"[\d.,\s-]+", h) for h in header)
    out = [f"[table {title}]".replace(" ]", "]")] if title else []
    if has_header:
        out.append("columns: " + " | ".join(header))
        for r in rows[1:]:
            cells = []
            for i, c in enumerate(r):
                c = str(c).strip()
                if not c:
                    continue
                name = header[i] if i < len(header) and header[i] else f"col{i + 1}"
                cells.append(f"{name}={c}")
            if cells:
                out.append("; ".join(cells))
    else:
        for r in rows:
            cells = [str(c).strip() for c in r if str(c).strip()]
            if cells:
                out.append(" | ".join(cells))
    return "\n".join(out)


# ---------------------------------------------------------------- office xml

def _open_office_zip(raw: bytes) -> zipfile.ZipFile:
    if raw[:8] == OLE_MAGIC:
        raise Unreadable("OLE container (encrypted or legacy Office file)")
    try:
        return zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as e:
        raise Unreadable(f"not a valid Office zip: {e}") from e


def _embedded_images(z: zipfile.ZipFile, prefix: str) -> list[Section]:
    out = []
    for name in z.namelist():
        if name.startswith(prefix) and Path(name).suffix.lower() in IMAGE_EXT:
            info = z.getinfo(name)
            if 2048 <= info.file_size <= 20 * 1024 * 1024:
                out.append(Section("", kind="image", image=z.read(name), label=f"embedded {Path(name).name}"))
    return out


def _docx_paragraph_text(p: ET.Element) -> str:
    parts = []
    for el in p.iter():
        t = _ns_strip(el.tag)
        if t == "t" and el.text:
            parts.append(el.text)
        elif t == "tab":
            parts.append("\t")
        elif t in ("br", "cr"):
            parts.append("\n")
    return "".join(parts)


def parse_docx(path: Path, raw: bytes) -> list[Section]:
    z = _open_office_zip(raw)
    names = z.namelist()
    if "word/document.xml" not in names:
        raise Unreadable("docx without word/document.xml")
    parts = ["word/document.xml"] + sorted(
        n for n in names if re.match(r"word/(header|footer|footnotes|endnotes|comments)\d*\.xml$", n)
    )
    blocks: list[str] = []
    for part in parts:
        root = ET.fromstring(z.read(part))
        body = next((el for el in root.iter() if _ns_strip(el.tag) == "body"), root)
        for el in body:
            t = _ns_strip(el.tag)
            if t == "p":
                s = _docx_paragraph_text(el).strip()
                if s:
                    blocks.append(s)
            elif t == "tbl":
                rows = []
                for tr in (x for x in el.iter() if _ns_strip(x.tag) == "tr"):
                    cells = []
                    for tc in (x for x in tr if _ns_strip(x.tag) == "tc"):
                        ps = [_docx_paragraph_text(p).strip() for p in tc.iter() if _ns_strip(p.tag) == "p"]
                        cells.append(" ".join(s for s in ps if s))
                    rows.append(cells)
                blocks.append(rows_to_text(rows))
            else:  # sdt blocks and friends: take their text flat
                s = " ".join(x.text for x in el.iter() if _ns_strip(x.tag) == "t" and x.text).strip()
                if s:
                    blocks.append(s)
    out = [Section("\n".join(blocks))] if blocks else []
    return out + _embedded_images(z, "word/media/")


def _col_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref or "A")
    n = 0
    for ch in letters.group(0) if letters else "A":
        n = n * 26 + ord(ch) - 64
    return n - 1


def parse_xlsx(path: Path, raw: bytes) -> list[Section]:
    z = _open_office_zip(raw)
    names = z.namelist()
    shared: list[str] = []
    if "xl/sharedStrings.xml" in names:
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root:
            shared.append("".join(t.text or "" for t in si.iter() if _ns_strip(t.tag) == "t"))
    # map sheet names to their part files
    sheet_files: list[tuple[str, str]] = []
    try:
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {r.get("Id"): r.get("Target") for r in rels}
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        for sh in (x for x in wb.iter() if _ns_strip(x.tag) == "sheet"):
            rid = next((v for k, v in sh.attrib.items() if k.endswith("}id")), None)
            target = rid_to_target.get(rid, "")
            target = target.lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            sheet_files.append((sh.get("name", target), target))
    except (KeyError, ET.ParseError):
        pass
    if not sheet_files:
        sheet_files = [(Path(n).stem, n) for n in sorted(names) if re.match(r"xl/worksheets/sheet\d+\.xml$", n)]

    sections = []
    for sheet_name, part in sheet_files:
        if part not in names:
            continue
        root = ET.fromstring(z.read(part))
        rows = []
        for row in (x for x in root.iter() if _ns_strip(x.tag) == "row"):
            cells: dict[int, str] = {}
            for c in (x for x in row if _ns_strip(x.tag) == "c"):
                typ = c.get("t", "")
                val = ""
                if typ == "inlineStr":
                    val = "".join(t.text or "" for t in c.iter() if _ns_strip(t.tag) == "t")
                else:
                    v = next((x for x in c if _ns_strip(x.tag) == "v"), None)
                    if v is not None and v.text is not None:
                        val = v.text
                        if typ == "s":
                            try:
                                val = shared[int(val)]
                            except (ValueError, IndexError):
                                pass
                        elif typ in ("", "n"):
                            val = _tidy_number(val)
                if val != "":
                    cells[_col_index(c.get("r", ""))] = val
            if cells:
                width = max(cells) + 1
                rows.append([cells.get(i, "") for i in range(width)])
        if rows:
            sections.append(Section(rows_to_text(rows, title=f"sheet {sheet_name}"), kind="table", label=f"sheet {sheet_name}"))
    return sections + _embedded_images(z, "xl/media/")


def _tidy_number(v: str) -> str:
    try:
        f = float(v)
    except ValueError:
        return v
    if f.is_integer() and abs(f) < 1e15:
        return str(int(f))
    return repr(round(f, 10)).rstrip("0").rstrip(".") if "e" not in repr(f) else v


def parse_pptx(path: Path, raw: bytes) -> list[Section]:
    z = _open_office_zip(raw)
    slides = sorted(
        (n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
        key=lambda n: int(re.search(r"(\d+)", Path(n).stem).group(1)),
    )
    out = []
    for i, n in enumerate(slides, 1):
        root = ET.fromstring(z.read(n))
        paras = []
        for p in (x for x in root.iter() if _ns_strip(x.tag) == "p"):
            s = "".join(t.text or "" for t in p.iter() if _ns_strip(t.tag) == "t").strip()
            if s:
                paras.append(s)
        if paras:
            out.append(Section("\n".join(paras), label=f"slide {i}"))
    return out + _embedded_images(z, "ppt/media/")


# ---------------------------------------------------------------- pdf

def parse_pdf(path: Path, raw: bytes) -> list[Section]:
    from pypdf import PdfReader  # imported lazily so a broken install cannot kill the walk

    if not raw.lstrip()[:5].startswith(b"%PDF"):
        raise Unreadable("no PDF header")
    reader = PdfReader(io.BytesIO(raw), strict=False)
    # An encrypted PDF is never a source, even if an empty password would open
    # it: the brief scores its contents as unanswerable.
    if reader.is_encrypted:
        raise Unreadable("encrypted PDF")
    out = []
    for i, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - one bad page must not kill the file
            text = ""
        text = text.strip()
        if text:
            out.append(Section(text, label=f"page {i}"))
        # Images on a page that carries little text are likely scans or
        # figures holding the answer; hand them to OCR.
        try:
            imgs = list(page.images) if len(text) < 400 else []
        except Exception:  # noqa: BLE001
            imgs = []
        for img in imgs[:4]:
            data = getattr(img, "data", b"")
            if data and len(data) >= 2048:
                out.append(Section("", kind="image", image=data, label=f"page {i} image"))
    return out


# ---------------------------------------------------------------- dispatch

def parse_file(path: Path, rel: str) -> ParsedFile:
    ext = path.suffix.lower()
    pf = ParsedFile(rel=rel, ext=ext)
    if ext in IMAGE_EXT:
        raw = read_bytes(path)
        if not raw:
            raise Unreadable("empty image")
        pf.sections = [Section("", kind="image", image=raw)]
        return pf
    if ext in TEXT_EXT:
        pf.sections = parse_text(path, read_bytes(path))
    elif ext in TABLE_EXT:
        pf.sections = parse_table(path, read_bytes(path))
    elif ext in PDF_EXT:
        pf.sections = parse_pdf(path, read_bytes(path))
    elif ext in DOCX_EXT:
        pf.sections = parse_docx(path, read_bytes(path))
    elif ext in XLSX_EXT:
        pf.sections = parse_xlsx(path, read_bytes(path))
    elif ext in PPTX_EXT:
        pf.sections = parse_pptx(path, read_bytes(path))
    else:
        raise Unreadable(f"no parser for {ext or 'extensionless'} file")
    return pf


def walk_corpus(root: Path):
    """Yield (path, rel) for every regular file, never raising.

    os.walk swallows directory errors via onerror; unreadable directories and
    broken symlinks are skipped rather than aborting the walk.
    """
    root = Path(root)
    errors: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: errors.append(str(e)), followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            try:
                st = p.stat()
            except OSError as e:
                errors.append(f"{p}: {e}")
                continue
            if not stat.S_ISREG(st.st_mode) or name.startswith("~$"):
                continue
            yield p, p.relative_to(root).as_posix()
