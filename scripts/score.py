#!/usr/bin/env python3
"""Grader-style run: exec app.py once per question, time it, score exact match.

    python3 scripts/score.py <app.py> <corpus> <sample-questions.json> [output_dir]
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

app, corpus, qfile = sys.argv[1], sys.argv[2], sys.argv[3]
out_dir = Path(sys.argv[4] if len(sys.argv) > 4 else os.environ.get("MC3_OUTPUT_DIR", "/app/output"))


def norm(s):
    return re.sub(r"[\s\-.·_]", "", str(s).upper())


qs = json.loads(Path(qfile).read_text())["queries"]
total, worst = 0, 0.0
for q in qs:
    qid = f"query_{q['n']:02d}"
    t = time.time()
    subprocess.run([sys.executable, app, "--corpus", corpus, "--query-id", qid, "--query", q["query"]], check=False)
    dt = time.time() - t
    worst = max(worst, dt)
    try:
        d = json.loads((out_dir / f"{qid}_output.json").read_text())
    except Exception as e:  # noqa: BLE001
        print(f"{qid} MALFORMED {e}")
        continue
    exp_a = [q["expected_answer"]] + q.get("answer_aliases", [])
    ok_a = any(norm(d.get("answer", "")) == norm(x) for x in exp_a)
    cites = {c.replace(corpus.rstrip("/") + "/", "") for c in d.get("citations", [])}
    ok_c = cites == set(q["expected_citations"])
    total += 20 if ok_a and ok_c else 0
    flag = "PASS" if ok_a and ok_c else ("WRONG-ANSWER" if not ok_a else "WRONG-CITES")
    print(f"{qid} {flag:12s} {dt:5.1f}s  answer={d.get('answer')!r} cites={sorted(cites)}"
          + ("" if ok_a and ok_c else f"   expected {q['expected_answer']!r} {q['expected_citations']}"))
print(f"SCORE {total}/200   slowest question {worst:.1f}s (limit 30s)")
