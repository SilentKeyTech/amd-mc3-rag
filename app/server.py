#!/usr/bin/env python3
"""Resident process: holds the model and the index for the life of the container.

Started by the container CMD. app.py is a thin client that talks to it over a
unix socket, so the model is loaded once, not once per question (trap 1).

Protocol: one JSON request line, one JSON response line, per connection.
  {"op": "ping"}
  {"op": "index", "corpus": "/app/corpus"}
  {"op": "query", "corpus": "/app/corpus", "query": "..."}
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.index import Store  # noqa: E402
from rag.pipeline import answer, build_index  # noqa: E402

SOCK = os.environ.get("MC3_SOCKET", "/tmp/mc3.sock")
INDEX_FILE = Path(os.environ.get("MC3_INDEX_DIR", "/app/index")) / "store.json"
START = float(os.environ.get("MC3_CONTAINER_START", time.time()))
STARTUP_BUDGET = float(os.environ.get("MC3_STARTUP_BUDGET", "540"))  # of the harness's 600 s

log = logging.getLogger("mc3.server")


class State:
    llm = None
    store: Store | None = None
    lock = threading.Lock()


def load_model():
    if os.environ.get("MC3_MOCK_LLM"):
        from rag.llm import MockLLM

        return MockLLM()
    from rag.llm import VLM

    return VLM()


def handle(req: dict) -> dict:
    op = req.get("op")
    if op == "ping":
        return {"ok": True, "model": State.llm is not None, "indexed": State.store is not None}
    if op == "index":
        corpus = Path(req.get("corpus", "/app/corpus"))
        store = build_index(corpus, State.llm, deadline=START + STARTUP_BUDGET)
        store.save(INDEX_FILE)
        State.store = store
        return {"ok": True, "files": len(store.files), "chunks": len(store.chunks)}
    if op == "query":
        corpus = Path(req.get("corpus", "/app/corpus"))
        if State.store is None or Path(State.store.corpus_root) != corpus:
            if INDEX_FILE.exists():
                State.store = Store.load(INDEX_FILE)
            if State.store is None or Path(State.store.corpus_root) != corpus:
                # never indexed: text-only index now (OCR would not fit in 30 s)
                State.store = build_index(corpus, None)
        if State.llm is None:
            return {"ok": True, "answer": "", "citations": [], "confidence": 0.0}
        a, c, conf = answer(State.store, State.llm, req.get("query", ""))
        return {"ok": True, "answer": a, "citations": c, "confidence": conf}
    return {"ok": False, "error": f"unknown op {op!r}"}


def serve_conn(conn: socket.socket) -> None:
    with conn:
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
        try:
            req = json.loads(buf.decode("utf-8") or "{}")
            if req.get("op") == "ping":
                resp = handle(req)
            else:
                with State.lock:  # one GPU job at a time
                    t0 = time.time()
                    resp = handle(req)
                    log.info("%s done in %.2fs: %s", req.get("op"), time.time() - t0, json.dumps(resp)[:300])
        except Exception as e:  # noqa: BLE001
            log.error("request failed: %s\n%s", e, traceback.format_exc())
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        try:
            conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
        except OSError:
            pass


def main() -> int:
    Path("/app/logs").mkdir(parents=True, exist_ok=True) if Path("/app").exists() else None
    handlers = [logging.StreamHandler(sys.stderr)]
    if Path("/app/logs").exists():
        handlers.append(logging.FileHandler("/app/logs/server.log"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", handlers=handlers)

    try:
        State.llm = load_model()
    except Exception as e:  # noqa: BLE001 - keep serving; queries will refuse rather than hang
        log.error("model load failed: %s\n%s", e, traceback.format_exc())
        State.llm = None
    if INDEX_FILE.exists():
        try:
            State.store = Store.load(INDEX_FILE)
            log.info("reloaded persisted index (%d chunks)", len(State.store.chunks))
        except Exception as e:  # noqa: BLE001
            log.warning("could not reload index: %s", e)

    try:
        os.unlink(SOCK)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK + ".tmp")
    srv.listen(16)
    os.replace(SOCK + ".tmp", SOCK)  # appear atomically, only once ready
    log.info("ready on %s after %.1fs", SOCK, time.time() - START)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=serve_conn, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
