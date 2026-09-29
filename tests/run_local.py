#!/usr/bin/env python3
"""Local, GPU-free test of everything except the model's judgement.

A mock VLM stands in for Qwen: OCR returns a fixed transcription per image, and
the answering call behaves like a SLOPPY model on purpose (cites every file it
was shown, cites the withdrawn datasheet, forgets the bridge file on the chain
question) so that the deterministic citation logic is what gets tested.

    python3 tests/run_local.py <corpus_dir> <sample-questions.json>
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from rag.llm import MockLLM  # noqa: E402
from rag.pipeline import answer, build_index, norm  # noqa: E402

corpus = Path(sys.argv[1])
qs = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["queries"]

OCR = {
    "specs/backplane_pinout.png": "TQ-40 backplane connector - pin assignment\nB11: GND\nB12: SDA\nB13: SCL\nB14: THERM_ALERT#\nB15: PRSNT#\nB16: GND\nTHERM_ALERT# is asserted low when the die exceeds the warning threshold. It is the only open-drain pin in the row.\nSheet 4 of 9 - mechanical drawings are in a separate package.",
    "support/asset_label.jpg": "Orrery Systems\nMODEL: TQ-40\nBOARD REVISION: REV-C2\nSN: OS4-118823-77\nMADE IN MALAYSIA",
}
ocr_by_bytes = {}
for rel, txt in OCR.items():
    try:
        ocr_by_bytes[(corpus / rel).read_bytes()] = txt
    except OSError:
        pass

EXPECT = {q["query"]: q for q in qs}
MODE = {"sloppy": True}


def responder(text: str, content) -> str:
    if text.rstrip().endswith("YES or NO."):
        return "YES"
    labels = dict(re.findall(r"===== \[(F\d+)\] file: (\S+)", text))
    q = re.search(r"Question: (.*)\nJSON:", text).group(1)
    exp = EXPECT[q]
    if not exp["expected_answer"]:
        return '{"evidence": "", "answer": "", "sources": []}'
    inv = {v: k for k, v in labels.items()}
    if exp["n"] == 9:  # forget the bridge file
        srcs = [inv[c] for c in exp["expected_citations"] if c in inv][-1:]
    else:  # cite everything shown
        srcs = list(labels)
    ans = exp["expected_answer"]
    if exp["n"] == 1:
        ans = "94 °C"
    if exp["n"] == 7:
        ans = "Pin B14"
    return json.dumps({"evidence": "...", "answer": ans, "sources": srcs})


llm = MockLLM(ocr_text=ocr_by_bytes, responder=responder)
t0 = time.time()
store = build_index(corpus, llm)
print(f"index: {len(store.files)} files, {len(store.chunks)} chunks, {time.time() - t0:.2f}s")
for f in store.files.values():
    print(f"  {f.status:7s} {'SUPERSEDED ' if f.superseded else ''}{f.rel}  {f.reason}")

score = 0
for q in qs:
    t = time.time()
    a, c, _ = answer(store, llm, q["query"])
    ok_a = norm(a) == norm(q["expected_answer"]) or any(norm(a) == norm(x) for x in q.get("answer_aliases", []))
    ok_c = set(c) == set(q["expected_citations"])
    score += 20 if ok_a and ok_c else 0
    print(f"Q{q['n']:>2} {'PASS' if ok_a and ok_c else 'FAIL'}  ans={a!r} cites={c}  ({time.time() - t:.3f}s)"
          + ("" if ok_a else f"  expected ans {q['expected_answer']!r}") + ("" if ok_c else f"  expected cites {q['expected_citations']}"))
print(f"score {score}/200")
