#!/bin/bash
# Real-GPU evaluation inside an AMD hackathon JupyterLab session (no Docker there).
# Same base image as the submission, so this exercises the real model on the real GPU.
#
#   bash scripts/notebook_eval.sh
#
# Model weights go to persistent storage so a new session does not re-download them.
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
PERSIST=/persistent; [ -d /persistent ] || PERSIST=/workspace
MODEL_DIR=${MODEL_DIR:-$PERSIST/models/qwen3-vl-8b}
WORK=${WORK:-/tmp/mc3eval}
mkdir -p "$WORK" "$PERSIST/models"

echo "=== 1. deps (torch pinned, trap 2)"
pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|pytorch-triton-rocm|numpy|pillow)==' > /tmp/constraints.txt
pip install -q -c /tmp/constraints.txt -r "$HERE/requirements.txt"
python3 -c "import torch; v=torch.__version__; print('torch', v, 'gpu', torch.cuda.get_device_name(0)); assert '+rocm' in v"

echo "=== 2. model -> $MODEL_DIR"
[ -f "$MODEL_DIR/config.json" ] || python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='$MODEL_DIR', allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])"

echo "=== 3. sample corpus with the hostile cases recreated"
cd "$WORK"
[ -d mc3-starter-kit ] || { curl -sSLO https://storage.googleapis.com/lablab-static-eu/share/mc3-starter-kit.zip; python3 -m zipfile -e mc3-starter-kit.zip .; }
mkdir -p mc3-starter-kit/mc3-corpus/archive
chmod 000 mc3-starter-kit/mc3-corpus/vendor/internal_audit.txt
[ "$(id -u)" = 0 ] && echo "NOTE: running as root, so the mode-000 file is still readable here; the Docker test covers it."

echo "=== 4. resident server (model loads once)"
export MC3_MODEL="$MODEL_DIR" MC3_SOCKET=/tmp/mc3eval.sock MC3_OUTPUT_DIR="$WORK/output" MC3_INDEX_DIR="$WORK/index"
export MC3_CONTAINER_START=$(date +%s) HF_HUB_OFFLINE=1
pkill -f "app/server.py" 2>/dev/null || true; rm -f $MC3_SOCKET
nohup python3 "$HERE/app/server.py" > "$WORK/server.log" 2>&1 &
( while sleep 1; do amd-smi metric --mem 2>/dev/null | grep -i used_vram || rocm-smi --showmeminfo vram 2>/dev/null | grep -i used; done ) > "$WORK/vram.log" 2>&1 &
VRAMMON=$!

t0=$(date +%s)
python3 "$HERE/app/app.py" --index "$WORK/mc3-starter-kit/mc3-corpus"
echo "startup + index: $(( $(date +%s) - t0 ))s (limit 600s)"
grep -E "loaded|warmup|indexed|skipped|OCR" "$WORK/server.log" | tail -20

echo "=== 5. ten questions, one process each, like the grader"
python3 "$HERE/scripts/score.py" "$HERE/app/app.py" "$WORK/mc3-starter-kit/mc3-corpus" "$WORK/mc3-starter-kit/sample-questions.json" "$WORK/output"

kill $VRAMMON 2>/dev/null || true
echo "=== VRAM samples (must stay 1-48 GiB)"; sort -u "$WORK/vram.log" | tail -5
echo "=== model outputs"; grep -E "model:|verifier:|refusing" "$WORK/server.log" | tail -30
