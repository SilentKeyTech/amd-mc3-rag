#!/usr/bin/env python3
"""Mini-Challenge 3 entry point. A thin, standard-library client.

    python3 /app/app.py --index /app/corpus
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."

The model and the index live in server.py, started by the container CMD, so a
question costs one socket round trip rather than a model load. Whatever happens
(server down, timeout, crash), a query ALWAYS writes a well-formed output file:
a refusal is scored, a missing file is not.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

T0 = time.time()
SOCK = os.environ.get("MC3_SOCKET", "/tmp/mc3.sock")
OUTPUT_DIR = Path(os.environ.get("MC3_OUTPUT_DIR", "/app/output"))
QUERY_BUDGET = float(os.environ.get("MC3_QUERY_BUDGET", "26"))     # harness allows 30 s
INDEX_WAIT = float(os.environ.get("MC3_INDEX_WAIT", "560"))        # harness allows 600 s startup


def call(req: dict, wait_for_socket: float, timeout: float) -> dict:
    deadline = time.time() + wait_for_socket
    while not os.path.exists(SOCK):
        if time.time() > deadline:
            raise TimeoutError("server socket never appeared")
        time.sleep(0.25)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    with s:
        while True:
            try:
                s.connect(SOCK)
                break
            except (ConnectionRefusedError, FileNotFoundError):
                if time.time() > deadline:
                    raise
                time.sleep(0.25)
        s.sendall((json.dumps(req) + "\n").encode("utf-8"))
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode("utf-8"))


def write_output(query_id: str, answer: str, citations: list[str], confidence: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{query_id}_output.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(
        json.dumps({"answer": answer, "citations": list(citations), "confidence": confidence}, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, out)  # never let the harness read a half-written file


def do_index(corpus: Path) -> int:
    try:
        resp = call({"op": "index", "corpus": str(corpus.resolve())}, INDEX_WAIT, INDEX_WAIT + 30)
        print(json.dumps(resp))
        if resp.get("ok"):
            return 0
    except Exception as e:  # noqa: BLE001
        print(f"index via server failed: {e}", file=sys.stderr)
    # Fallback: persist a text-only index in-process so queries still have one.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rag.pipeline import build_index

    store = build_index(corpus, None)
    store.save(Path(os.environ.get("MC3_INDEX_DIR", "/app/index")) / "store.json")
    print(f"fallback text-only index: {len(store.chunks)} chunks", file=sys.stderr)
    return 0


def do_query(corpus: Path, query_id: str, query: str) -> int:
    answer, citations, confidence = "", [], 0.0
    try:
        remaining = QUERY_BUDGET - (time.time() - T0)
        resp = call(
            {"op": "query", "corpus": str(corpus.resolve()), "query": query},
            wait_for_socket=min(5.0, remaining),
            timeout=max(1.0, remaining),
        )
        if resp.get("ok"):
            answer = str(resp.get("answer") or "")
            citations = [str(c) for c in resp.get("citations") or []] if answer else []
            confidence = float(resp.get("confidence") or 0.0)
        else:
            print(f"server error: {resp.get('error')}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 - always fall through to a valid file
        print(f"query failed: {e}", file=sys.stderr)
    write_output(query_id, answer, citations, confidence)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, help="build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem the harness assigns, e.g. query_01")
    ap.add_argument("--query", help="the question to answer")
    args = ap.parse_args()

    if args.index is not None:
        return do_index(args.index)
    if not args.query_id:
        ap.error("a query needs --query-id")
    return do_query(args.corpus or Path("/app/corpus"), args.query_id, args.query or "")


if __name__ == "__main__":
    raise SystemExit(main())
