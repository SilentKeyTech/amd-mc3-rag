"""Chunk store and lexical retrieval (BM25), persisted as one JSON file.

The corpus is small enough that exact lexical scoring over chunks is fast and,
for this task, more precise than dense embeddings: questions name part numbers,
ticket ids, error codes and constants, which BM25 matches exactly.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

STOP = set(
    """a an the of to in on for and or is are was were be been by with at from as that this which what
    who whom whose when where why how does do did it its into than then there their they them these those
    per via any all each our your his her not no yes can could should would will shall may might must
    about under over after before between during without within upon also only such same other""".split()
)

COMPOUND = re.compile(r"[a-z0-9]+(?:[-_./#][a-z0-9]+)*#?")
IDENT = re.compile(r"\b(?=[A-Za-z0-9_#-]*\d)(?=[A-Za-z0-9_#-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9_#-]{2,}\b|\b[A-Z][A-Z0-9]*_[A-Z0-9_#]+\b")

SUPERSEDED_NAME = re.compile(r"(withdrawn|superseded|obsolete|deprecated|_old\b|\bold_|backup|\.bak)", re.I)


def tokenize(text: str) -> list[str]:
    toks = []
    for m in COMPOUND.finditer(text.lower()):
        w = m.group(0).rstrip("#")
        if not w:
            continue
        parts = re.split(r"[-_./#]", w)
        if len(parts) > 1:
            toks.append(w)
        for p in parts:
            if p and p not in STOP and (len(p) > 1 or p.isdigit()):
                toks.append(p)
    return toks


def identifiers(text: str) -> set[str]:
    """Id-like tokens: ORR-1847, E7731, THERM_ALERT#, 4.3.2, ORR-FAN-2214-B."""
    out = {m.group(0).rstrip("#").upper() for m in IDENT.finditer(text)}
    return {x for x in out if len(x) >= 3 and not re.fullmatch(r"\d{1,4}", x)}


@dataclass
class Chunk:
    id: int
    file: str
    text: str
    kind: str = "text"      # text | table | image
    label: str = ""


@dataclass
class FileMeta:
    rel: str
    ext: str
    status: str             # ok | skipped
    reason: str = ""
    superseded: bool = False
    is_image: bool = False
    full_text: str = ""     # everything we could read, for grounding checks


@dataclass
class Store:
    chunks: list[Chunk] = field(default_factory=list)
    files: dict[str, FileMeta] = field(default_factory=dict)
    corpus_root: str = ""

    # BM25 state, rebuilt on load
    def build(self) -> None:
        self._tf = []
        self._len = []
        df: Counter[str] = Counter()
        for c in self.chunks:
            toks = tokenize(c.file.replace("/", " ") + " " + c.text)
            tf = Counter(toks)
            self._tf.append(tf)
            self._len.append(len(toks) or 1)
            df.update(tf.keys())
        n = max(len(self.chunks), 1)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._avg = sum(self._len) / n if self._len else 1.0
        # file-level document frequency of identifiers, for bridge detection
        self.ident_df: Counter[str] = Counter()
        for f in self.files.values():
            if f.status == "ok":
                self.ident_df.update(identifiers(f.full_text))

    def bm25(self, query: str, k1: float = 1.3, b: float = 0.6) -> list[tuple[float, int]]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            norm = k1 * (1 - b + b * self._len[i] / self._avg)
            for t in q:
                f = tf.get(t)
                if f:
                    s += self._idf.get(t, 0.0) * f * (k1 + 1) / (f + norm)
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return scores

    def chunks_with_identifier(self, ident: str) -> list[int]:
        pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(ident) + r"(?![A-Za-z0-9])", re.I)
        return [c.id for c in self.chunks if pat.search(c.text)]

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "corpus_root": self.corpus_root,
                    "chunks": [asdict(c) for c in self.chunks],
                    "files": {k: asdict(v) for k, v in self.files.items()},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Store":
        d = json.loads(path.read_text(encoding="utf-8"))
        s = cls(
            chunks=[Chunk(**c) for c in d["chunks"]],
            files={k: FileMeta(**v) for k, v in d["files"].items()},
            corpus_root=d.get("corpus_root", ""),
        )
        s.build()
        return s


def split_text(text: str, target: int = 1100, overlap: int = 200) -> list[str]:
    text = text.strip()
    if len(text) <= target * 1.3:
        return [text] if text else []
    lines = text.splitlines()
    out, cur, size = [], [], 0
    for ln in lines:
        while len(ln) > target:  # very long single lines (minified text)
            head, ln = ln[:target], ln[target - overlap:]
            if cur:
                out.append("\n".join(cur))
                cur, size = [], 0
            out.append(head)
        cur.append(ln)
        size += len(ln) + 1
        if size >= target:
            out.append("\n".join(cur))
            # carry the tail forward as overlap
            tail, tsize = [], 0
            for l2 in reversed(cur):
                if tsize + len(l2) > overlap:
                    break
                tail.insert(0, l2)
                tsize += len(l2) + 1
            cur, size = tail, tsize
    if cur and (not out or "\n".join(cur) not in out[-1]):
        out.append("\n".join(cur))
    return out


def split_table(text: str, rows_per_chunk: int = 15) -> list[str]:
    lines = text.splitlines()
    head = [l for l in lines[:2] if l.startswith(("[table", "columns:"))]
    body = lines[len(head):]
    if len(body) <= rows_per_chunk * 1.5:
        return [text]
    return ["\n".join(head + body[i:i + rows_per_chunk]) for i in range(0, len(body), rows_per_chunk)]


def is_superseded(rel: str, text: str) -> bool:
    if SUPERSEDED_NAME.search(Path(rel).name):
        return True
    # Only the document's own title block counts: a current revision often
    # says "revision 1 is withdrawn" further down, which must not flag it.
    head = "\n".join([l for l in text.splitlines() if l.strip()][:4])
    return bool(re.search(r"\b(withdrawn|superseded by|obsolete|deprecated|do not use)\b", head, re.I))
