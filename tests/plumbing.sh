#!/bin/sh
# Client/server plumbing test with the mock model (no GPU). Usage: plumbing.sh <python> <corpus>
PY=${1:-python3}; CORPUS=${2:-/app/corpus}
HERE=$(cd "$(dirname "$0")/.." && pwd)
export MC3_MOCK_LLM=1 MC3_SOCKET=/tmp/t.sock MC3_OUTPUT_DIR=/tmp/mc3o MC3_INDEX_DIR=/tmp/mc3i
rm -rf /tmp/mc3o /tmp/mc3i /tmp/t.sock
$PY "$HERE/app/server.py" >/tmp/srv.log 2>&1 &
SRV=$!
echo "--- index"; $PY "$HERE/app/app.py" --index "$CORPUS"
echo "--- query with server up"
start=$(date +%s.%N)
$PY "$HERE/app/app.py" --corpus "$CORPUS" --query-id query_01 --query "What is the max junction temperature?"
echo "elapsed $(echo "$(date +%s.%N) - $start" | bc)"; cat /tmp/mc3o/query_01_output.json; echo
kill $SRV; sleep 1; rm -f /tmp/t.sock
echo "--- query with server DOWN (must still write a refusal)"
MC3_QUERY_BUDGET=3 $PY "$HERE/app/app.py" --corpus "$CORPUS" --query-id query_02 --query "x"
cat /tmp/mc3o/query_02_output.json; echo
echo "--- persisted index"; ls -la /tmp/mc3i
echo "--- server log"; tail -4 /tmp/srv.log
