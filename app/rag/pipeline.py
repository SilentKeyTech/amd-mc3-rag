"""Index and answer. The model proposes; deterministic code checks.

answer() flow:
  1. BM25 over chunks, then one identifier hop: ids found on the query-relevant
     lines of the top hits (a ticket, an error code) pull in the files that
     define them. This is how "log -> ticket -> bug database" chains are found.
  2. The VLM reads the candidate files (images are shown as images) and returns
     a value plus the file labels it used.
  3. Code enforces the rules the grader scores:
       - the value must actually occur in a readable file (no recall, no guess)
       - exactly one answer-bearing file, preferring current over superseded
       - a second file survives only as a genuine bridge: it holds an
         identifier that sits next to the answer in the answer file and that
         the question itself did not supply
  4. A short YES/NO verification pass rejects near-miss answers (a related
     item's value, a different quantity) and turns them into refusals.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from .index import Chunk, FileMeta, Store, identifiers, is_superseded, split_table, split_text, tokenize
from .parsers import Unreadable, parse_file, walk_corpus

log = logging.getLogger("mc3.pipeline")

MAX_CONTEXT_FILES = 7
MAX_CONTEXT_CHARS = 14000
MAX_IMAGES_SHOWN = 2

# ====================================================================== index


def build_index(corpus: Path, llm, deadline: float | None = None) -> Store:
    t0 = time.time()
    corpus = Path(corpus)
    store = Store(corpus_root=str(corpus))
    pending_images: list[tuple[str, object]] = []  # (rel, Section)
    file_texts: dict[str, list[str]] = {}

    for path, rel in walk_corpus(corpus):
        try:
            pf = parse_file(path, rel)
        except PermissionError as e:
            store.files[rel] = FileMeta(rel, path.suffix.lower(), "skipped", f"permission: {e}")
            continue
        except Unreadable as e:
            store.files[rel] = FileMeta(rel, path.suffix.lower(), "skipped", str(e))
            continue
        except Exception as e:  # noqa: BLE001 - never let one file stop the walk
            store.files[rel] = FileMeta(rel, path.suffix.lower(), "skipped", f"{type(e).__name__}: {e}")
            continue
        meta = FileMeta(rel, pf.ext, "ok", is_image=all(s.kind == "image" for s in pf.sections) and bool(pf.sections))
        store.files[rel] = meta
        file_texts[rel] = []
        for sec in pf.sections:
            if sec.kind == "image":
                pending_images.append((rel, sec))
                continue
            _add_chunks(store, rel, sec.text, sec.kind, sec.label)
            file_texts[rel].append(sec.text)

    # OCR last, so a slow or failing vision pass can never cost text files.
    for rel, sec in pending_images:
        if deadline and time.time() > deadline:
            log.warning("index deadline reached; skipping OCR of %s", rel)
            continue
        try:
            text = llm.ocr(sec.image) if llm is not None else ""
        except Exception as e:  # noqa: BLE001
            log.warning("OCR failed for %s: %s", rel, e)
            text = ""
        text = (text or "").strip()
        if text:
            label = sec.label or "image"
            _add_chunks(store, rel, f"[text read from {label}]\n{text}", "image", sec.label)
            file_texts[rel].append(text)

    for rel, texts in file_texts.items():
        meta = store.files[rel]
        meta.full_text = "\n".join(texts)
        meta.superseded = is_superseded(rel, meta.full_text)
        if not texts:
            meta.status, meta.reason = "skipped", "no readable content"

    store.build()
    ok = sum(1 for f in store.files.values() if f.status == "ok")
    log.info("indexed %d files (%d skipped), %d chunks in %.1fs", ok, len(store.files) - ok, len(store.chunks), time.time() - t0)
    for f in store.files.values():
        if f.status != "ok":
            log.info("  skipped %s: %s", f.rel, f.reason)
    return store


def _add_chunks(store: Store, rel: str, text: str, kind: str, label: str) -> None:
    pieces = split_table(text) if kind == "table" else split_text(text)
    for p in pieces:
        if p.strip():
            store.chunks.append(Chunk(id=len(store.chunks), file=rel, text=p, kind=kind, label=label))


# ================================================================== retrieval


def _line_overlap(line: str, qtoks: set[str]) -> int:
    return len(qtoks & set(tokenize(line)))


def retrieve(store: Store, query: str) -> tuple[list[str], dict[str, list[int]], dict[str, tuple[str, str]]]:
    """Return (ordered files, chunk ids per file, hop_from[file] = (source file, identifier))."""
    qtoks = set(tokenize(query))
    q_ids = identifiers(query)
    scored = store.bm25(query)
    by_file: dict[str, list[int]] = {}
    order: list[str] = []
    for _, cid in scored:
        f = store.chunks[cid].file
        if store.files[f].status != "ok":
            continue
        if f not in by_file:
            if len(order) >= MAX_CONTEXT_FILES:
                continue
            order.append(f)
            by_file[f] = []
        if len(by_file[f]) < 3:
            by_file[f].append(cid)

    # identifier hop from the query-relevant lines of the top files
    hop_from: dict[str, tuple[str, str]] = {}
    hop_ids: list[tuple[str, str]] = []
    for f in order[:4]:
        lines = []
        for cid in by_file[f]:
            lines += store.chunks[cid].text.splitlines()
        ranked = sorted(((_line_overlap(l, qtoks), l) for l in lines), key=lambda x: -x[0])
        for ov, line in ranked[:6]:
            if ov == 0:
                break
            for ident in identifiers(line):
                if ident in q_ids or any(ident in q or q in ident for q in q_ids):
                    continue
                if 1 <= store.ident_df.get(ident, 0) <= 4:
                    hop_ids.append((ident, f))
    seen = set()
    for ident, src in hop_ids:
        if ident in seen:
            continue
        seen.add(ident)
        for cid in store.chunks_with_identifier(ident):
            f = store.chunks[cid].file
            if f == src or store.files[f].status != "ok":
                continue
            if f not in by_file:
                order.append(f)
                by_file[f] = []
                hop_from[f] = (src, ident)
            if cid not in by_file[f] and len(by_file[f]) < 3:
                by_file[f].insert(0, cid)
        if len(order) >= MAX_CONTEXT_FILES + 3:
            break
    return order, by_file, hop_from


# ==================================================================== prompts

SYSTEM = (
    "You are a precise document question-answering system. You answer ONLY from the numbered source "
    "files you are given, never from memory: the products and companies in these files are fictional, so "
    "anything you remember about them is wrong. You reply with a single JSON object and nothing else."
)

RULES = """Rules:
1. The answer is the VALUE ONLY: a number, a part number, a version, a quarter, a code, a name. No sentence, no explanation, no restating the question.
2. Give the COMPLETE value as printed in the source. Keep qualifiers that identify it: a quarter keeps its fiscal year ("Q3 FY27", not "Q3"); a revision keeps its prefix ("REV-C2", not "C2"); a part number keeps every segment. For a plain measured number you may drop the unit.
3. If a source is marked SUPERSEDED/WITHDRAWN, do not answer from it when a current source covers the same fact.
4. Make sure the value is about EXACTLY the item and quantity the question asks about, not a related component, a different product, a different volume tier, or an old/legacy value. If the sources do not state it explicitly, the answer is "".
5. "sources" lists ONLY the files you actually needed: a file belongs there only if removing it would make the answer impossible. A file that merely discusses the same topic or repeats the value is NOT a source. If one file gave you an identifier (ticket, error code, part number) and another file gave the value for that identifier, list both.
6. If the sources do not contain the answer, reply {"evidence": "", "answer": "", "sources": []}. Never guess.

Reply with JSON only, in this shape:
{"evidence": "<the exact short line(s) from the source(s) that give the answer>", "answer": "<value or empty>", "sources": ["F1"]}"""


def _file_block(store: Store, label: str, rel: str, cids: list[int]) -> str:
    meta = store.files[rel]
    tag = "  [SUPERSEDED/WITHDRAWN DOCUMENT]" if meta.superseded else ""
    kind = "  [image; its printed text is transcribed below and the image itself is attached]" if meta.is_image else ""
    body = "\n...\n".join(store.chunks[c].text for c in sorted(cids))
    return f"===== [{label}] file: {rel}{tag}{kind}\n{body}\n"


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    ans = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
    srcs = re.findall(r"F\d+", text.split('"sources"', 1)[1]) if '"sources"' in text else []
    return {"answer": ans.group(1) if ans else "", "sources": srcs}


# ============================================================ answer cleanup

_NORM_DROP = re.compile(r"[\s\-.·_]")
UNIT = (
    r"(?:°\s*[CF]|deg(?:rees?)?\s*[CF]?|[CFK]|W|kW|mW|V|mV|A|mA|Hz|kHz|MHz|GHz|s|sec|secs|seconds?|ms|us|µs|"
    r"min|minutes?|h|hrs?|hours?|days?|weeks?|months?|years?|%|percent|mm|cm|m|km|in|inch(?:es)?|g|kg|lb|"
    r"B|KB|MB|GB|TB|KiB|MiB|GiB|TiB|Gbps|Mbps|GB/s|TB/s|MB/s|units?|pcs|pieces|USD|EUR|dollars?)"
)
PREFIX = re.compile(r"^(?:the\s+)?(?:pin|version|firmware(?:\s+version)?|release|ticket|error\s+code|code|part\s+(?:number|no\.?)|p/n|value|answer)\s*[:#]?\s+", re.I)


def norm(s: str) -> str:
    return _NORM_DROP.sub("", s.upper())


def clean_answer(ans: str) -> str:
    a = str(ans or "").strip().strip("`*\"'“”‘’").strip()
    a = re.sub(r"^(?:the\s+answer\s+is|answer)\s*[:\-]?\s*", "", a, flags=re.I)
    a = a.rstrip(".;,").strip()
    if a.lower() in {"", "none", "null", "n/a", "na", "unknown", "not found", "not available", "not stated", "not specified"}:
        return ""
    if re.search(r"\b(i (?:do not|don't|could not|cannot)|not (?:in|found|mentioned|provided|available)|no information)\b", a, re.I):
        return ""
    a = PREFIX.sub("", a).strip()
    m = re.fullmatch(r"([$€£]?\s*[-+]?\d[\d,]*(?:\.\d+)?)\s*" + UNIT + r"\.?", a, re.I)
    if m:
        a = m.group(1).replace("$", "").replace("€", "").replace("£", "").strip()
    elif re.fullmatch(r"[$€£]\s*\d[\d,]*(?:\.\d+)?", a):
        a = a[1:].strip()
    return a


def contains_value(text: str, ans: str) -> bool:
    """Normalized containment with digit boundaries (so '94' does not match '1947')."""
    n_text, n_ans = norm(text), norm(ans)
    if not n_ans:
        return False
    variants = {n_ans, n_ans.replace(",", "")}
    hay = {n_text, n_text.replace(",", "")}
    for v in variants:
        for h in hay:
            start = 0
            while True:
                i = h.find(v, start)
                if i < 0:
                    break
                before = h[i - 1] if i > 0 else ""
                after = h[i + len(v)] if i + len(v) < len(h) else ""
                ok_b = not (v[0].isdigit() and before.isdigit())
                ok_a = not (v[-1].isdigit() and after.isdigit())
                if ok_b and ok_a:
                    return True
                start = i + 1
    return False


def _answer_lines(text: str, ans: str, window: int = 3) -> str:
    lines = text.splitlines()
    hits = [i for i, l in enumerate(lines) if contains_value(l, ans)]
    keep = set()
    for i in hits:
        keep.update(range(max(0, i - window), min(len(lines), i + window + 1)))
    return "\n".join(lines[i] for i in sorted(keep))


# ===================================================================== answer


def answer(store: Store, llm, query: str) -> tuple[str, list[str], float]:
    order, by_file, hop_from = retrieve(store, query)
    if not order:
        return "", [], 0.0

    # assemble context within budget, images attached inline
    labels: dict[str, str] = {}
    content: list[dict] = []
    used = 0
    images_shown: set[str] = set()
    for rel in order:
        block = _file_block(store, f"F{len(labels) + 1}", rel, by_file[rel])
        if used + len(block) > MAX_CONTEXT_CHARS and labels:
            continue
        label = f"F{len(labels) + 1}"
        labels[label] = rel
        used += len(block)
        content.append({"type": "text", "text": block})
        if store.files[rel].is_image and len(images_shown) < MAX_IMAGES_SHOWN:
            raw = _read_image(store, rel)
            if raw:
                content.append({"type": "image", "image": raw})
                images_shown.add(rel)
    content.append({"type": "text", "text": f"\n{RULES}\n\nQuestion: {query}\nJSON:"})

    raw_out = llm.generate(content, max_new_tokens=200, system=SYSTEM)
    log.info("model: %s", raw_out.replace("\n", " ")[:400])
    parsed = _parse_json(raw_out)
    ans = clean_answer(parsed.get("answer", ""))
    if not ans:
        return "", [], 0.0
    srcs = parsed.get("sources") or []
    if isinstance(srcs, str):
        srcs = re.findall(r"F\d+", srcs)
    cited = []
    for s in srcs:
        m = re.search(r"F\d+", str(s))
        rel = labels.get(m.group(0)) if m else None
        if rel is None and str(s) in store.files:
            rel = str(s)
        if rel and rel not in cited:
            cited.append(rel)

    cites = select_citations(store, query, ans, cited, list(labels.values()), images_shown, hop_from)
    if not cites:
        log.info("refusing: answer %r is not grounded in any readable file", ans)
        return "", [], 0.0

    if not verify(store, llm, query, ans, cites, images_shown):
        log.info("refusing: verifier rejected %r from %s", ans, cites)
        return "", [], 0.0
    return ans, cites, 0.8


def _grounded(store: Store, rel: str, ans: str, images_shown: set[str]) -> bool:
    return contains_value(store.files[rel].full_text, ans) or rel in images_shown


def select_citations(store, query, ans, cited, context_files, images_shown, hop_from) -> list[str]:
    q_ids = identifiers(query)
    qtoks = set(tokenize(query))

    # 1. the single answer-bearing file
    bearing_cited = [f for f in cited if _grounded(store, f, ans, images_shown)]
    bearing_ctx = [f for f in context_files if _grounded(store, f, ans, images_shown)]
    primary = None
    for pool in (bearing_cited, bearing_ctx):
        current = [f for f in pool if not store.files[f].superseded]
        if current:
            primary = current[0]
            break
    if primary is None:
        pool = bearing_cited or bearing_ctx
        if not pool:
            return []  # value occurs nowhere we can read: recalled or guessed
        primary = pool[0]

    # 2. bridge files: hold an identifier that sits next to the answer in the
    #    primary file and that the question did not supply
    #    ("next to" = on the answer's own line; for tables, its own row), and
    #    rare enough to be a key rather than a product name used everywhere
    near = _answer_lines(store.files[primary].full_text, ans, window=0)
    n_ans = norm(ans)
    bridge_ids = {
        i for i in identifiers(near)
        if i not in q_ids and norm(i) not in n_ans and n_ans not in norm(i)
        and not any(norm(i) in norm(q) or norm(q) in norm(i) for q in q_ids)
        and store.ident_df.get(i, 0) <= 3
    }

    def bridge_id(f: str) -> str | None:
        if f == primary or store.files[f].superseded:
            return None
        ids = identifiers(store.files[f].full_text)
        hits = [i for i in bridge_ids if i in ids]
        return hits[0] if hits else None

    cites = [primary]
    for f in cited:
        if f != primary and bridge_id(f):
            cites.append(f)
    if len(cites) == 1:
        # the model may name only the value file on a chain question; add the
        # file the chain started from when the hop is unambiguous
        if primary in hop_from and bridge_id(hop_from[primary][0]):
            cites.append(hop_from[primary][0])
        else:
            best = None
            for f in context_files:
                bid = bridge_id(f)
                if not bid or f == primary:
                    continue
                # how directly does the question hit the answer row itself?
                prim_rows = [l for l in near.splitlines() if bid in identifiers(l)]
                prim_ov = max((_line_overlap(l, qtoks) for l in prim_rows), default=0)
                lines = store.files[f].full_text.splitlines()
                for i, l in enumerate(lines):
                    if bid in identifiers(l):
                        win = " ".join(lines[max(0, i - 2): i + 3])
                        ov = _line_overlap(win, qtoks)
                        if ov >= 2 and ov > prim_ov and (best is None or ov > best[0]):
                            best = (ov, f)
            if best:
                cites.append(best[1])
    return cites[:3]


VERIFY = """Question: {q}
Proposed answer: {a}

Using ONLY the source text above, is the proposed answer explicitly stated as the answer to exactly this question, for exactly the item, product, quantity and condition the question asks about (not a related part, a different product, a different volume or tier, or a withdrawn value)? If a chain of sources is used, every step must be explicit.
Reply with exactly one word: YES or NO."""


def verify(store: Store, llm, query: str, ans: str, cites: list[str], images_shown: set[str]) -> bool:
    content = []
    for rel in cites:
        meta = store.files[rel]
        text = meta.full_text
        if len(text) > 5000:
            focus = _answer_lines(text, ans, window=6)
            # a bridge file may not contain the answer; keep its id lines instead
            text = focus if focus else text[:5000]
        content.append({"type": "text", "text": f"===== file: {rel}\n{text[:6000]}\n"})
        if rel in images_shown:
            raw = _read_image(store, rel)
            if raw:
                content.append({"type": "image", "image": raw})
    content.append({"type": "text", "text": VERIFY.format(q=query, a=ans)})
    try:
        out = llm.generate(content, max_new_tokens=3)
    except Exception as e:  # noqa: BLE001 - a failed verifier must not cost a good answer
        log.warning("verifier failed: %s", e)
        return True
    log.info("verifier: %s", out)
    return not out.strip().upper().startswith("NO")


def _read_image(store: Store, rel: str) -> bytes | None:
    try:
        return (Path(store.corpus_root) / rel).read_bytes()
    except OSError:
        return None
