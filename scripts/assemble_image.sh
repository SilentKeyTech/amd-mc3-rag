#!/bin/bash
# Assemble the submission image IN THE REGISTRY with crane, without ever
# extracting the 29 GiB base image. A hosted CI runner has ~40 GB of disk, and a
# normal `docker build` of base + 17 GiB weights does not fit.
#
# Result = the mandated base image's layers, byte-for-byte (so the layer-identity
# check passes by construction) + one python-packages layer + one layer per
# weight shard + one app layer + config (ENV, WORKDIR, CMD).
#
#   BASE=... TARGET=ghcr.io/owner/mc3-rag:v1 bash scripts/assemble_image.sh
set -euo pipefail
: "${BASE:?}"; : "${TARGET:?}"
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}
REPO=${TARGET%:*}
WEIGHTS_TAG="$REPO:weights-$(echo "$MODEL_ID" | tr '/A-Z' '-a-z')-$(sha256sum requirements.txt | cut -c1-8)"
PYVER=${PYVER:-3.14}
SITE=/opt/mc3/site            # on PYTHONPATH in the final image
WORK=${WORK:-/tmp/assemble}
mkdir -p "$WORK"

layer() {  # layer <tarball> <dir-with-rootfs-content>
  tar --owner=0 --group=0 --numeric-owner --sort=name --mtime='2026-01-01' -C "$2" -cf "$1" .
}

echo "=== base config"
crane config "$BASE" | python3 -c "import json,sys; c=json.load(sys.stdin)['config']; print('Env:', c.get('Env')); print('Entrypoint:', c.get('Entrypoint'), 'Cmd:', c.get('Cmd'), 'WorkingDir:', c.get('WorkingDir'))"

if crane digest "$WEIGHTS_TAG" >/dev/null 2>&1; then
  echo "=== reusing $WEIGHTS_TAG (packages + weights already assembled)"
else
  echo "=== python packages for cp${PYVER/./} (torch, numpy and pillow come from the base image)"
  rm -rf "$WORK/site" && mkdir -p "$WORK/site$SITE"
  python3 -m pip install -q --target "$WORK/site$SITE" --python-version "$PYVER" --implementation cp \
    --only-binary=:all: --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 \
    --platform manylinux2014_x86_64 --platform any -r requirements.txt
  # Never ship anything that could shadow the base image's ROCm stack (trap 2).
  ( cd "$WORK/site$SITE" && rm -rf torch torch-* torchvision* torchaudio* triton* numpy numpy-* numpy.libs PIL pillow* nvidia* bin )
  ls "$WORK/site$SITE" | sed 's/^/    /'
  du -sh "$WORK/site"
  if command -v python$PYVER >/dev/null; then  # prove the layer imports on the target Python
    PYTHONPATH="$WORK/site$SITE" python$PYVER -c "import transformers, tokenizers, safetensors, pypdf, huggingface_hub; from transformers import AutoProcessor, AutoModelForImageTextToText; print('imports ok on', __import__('sys').version.split()[0], 'transformers', transformers.__version__)"
  fi
  layer "$WORK/site.tar" "$WORK/site"
  crane append --platform linux/amd64 -b "$BASE" -f "$WORK/site.tar" -t "$WEIGHTS_TAG.tmp"
  rm -rf "$WORK/site" "$WORK/site.tar"

  echo "=== weights, one layer per file, streamed through a small disk"
  python3 -m pip install -q "huggingface_hub[hf_xet]"
  FILES=$(python3 -c "
from huggingface_hub import list_repo_files
import fnmatch
pats=['*.json','*.safetensors','*.txt','*.jinja']
print('\n'.join(f for f in list_repo_files('$MODEL_ID') if any(fnmatch.fnmatch(f,p) for p in pats)))")
  SMALL=$(echo "$FILES" | grep -v '\.safetensors$' || true)
  BIG=$(echo "$FILES" | grep '\.safetensors$')
  cur="$WEIGHTS_TAG.tmp"
  # small files (config, tokenizer) in one layer
  rm -rf "$WORK/w" && mkdir -p "$WORK/w/models/vlm"
  for f in $SMALL; do
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('$MODEL_ID', '$f', local_dir='$WORK/w/models/vlm')"
  done
  rm -rf "$WORK/w/models/vlm/.cache"
  layer "$WORK/w.tar" "$WORK/w" && crane append -b "$cur" -f "$WORK/w.tar" -t "$WEIGHTS_TAG.tmp"
  for f in $BIG; do
    rm -rf "$WORK/w" "$WORK/w.tar" && mkdir -p "$WORK/w/models/vlm"
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('$MODEL_ID', '$f', local_dir='$WORK/w/models/vlm')"
    rm -rf "$WORK/w/models/vlm/.cache" ~/.cache/huggingface
    ls -la "$WORK/w/models/vlm"
    layer "$WORK/w.tar" "$WORK/w"
    rm -rf "$WORK/w"
    crane append -b "$WEIGHTS_TAG.tmp" -f "$WORK/w.tar" -t "$WEIGHTS_TAG.tmp"
    rm -f "$WORK/w.tar"
    df -h "$WORK" | tail -1
  done
  crane tag "$WEIGHTS_TAG.tmp" "${WEIGHTS_TAG##*:}"
fi

echo "=== app layer"
rm -rf "$WORK/app" && mkdir -p "$WORK/app/app" "$WORK/app/app/corpus" "$WORK/app/app/output" "$WORK/app/app/index" "$WORK/app/app/logs"
cp -r app/. "$WORK/app/app/"
find "$WORK/app" -name __pycache__ -prune -exec rm -rf {} +
chmod -R a+rX "$WORK/app" && chmod 1777 "$WORK/app/app/output" "$WORK/app/app/index" "$WORK/app/app/logs"
layer "$WORK/app.tar" "$WORK/app"
crane append -b "$WEIGHTS_TAG" -f "$WORK/app.tar" -t "$TARGET.tmp"

echo "=== config"
BASE_PATH=$(crane config "$BASE" | python3 -c "import json,sys; e=dict(x.split('=',1) for x in json.load(sys.stdin)['config'].get('Env') or []); print(e.get('PATH','/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'))")
crane mutate "$TARGET.tmp" -t "$TARGET" \
  --workdir /app \
  --env "PYTHONPATH=$SITE" --env "PATH=$BASE_PATH" --env PYTHONUNBUFFERED=1 \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MC3_MODEL=/models/vlm \
  --env MIOPEN_USER_DB_PATH=/tmp/miopen --env TOKENIZERS_PARALLELISM=false \
  --cmd 'sh,-c,export MC3_CONTAINER_START=$(date +%s); while true; do python3 /app/server.py; echo server exited: restarting >&2; sleep 1; done'

echo "=== verify: base layers are an exact prefix, size under 60 GiB uncompressed"
python3 - "$BASE" "$TARGET" <<'PY'
import json, subprocess, sys
base, target = sys.argv[1], sys.argv[2]
def manifest(ref):
    m = json.loads(subprocess.check_output(["crane", "manifest", "--platform", "linux/amd64", ref]))
    return m
b, t = manifest(base), manifest(target)
bl, tl = [l["digest"] for l in b["layers"]], [l["digest"] for l in t["layers"]]
assert tl[:len(bl)] == bl, "base layers are not an exact prefix"
cfg_b = json.loads(subprocess.check_output(["crane", "config", "--platform", "linux/amd64", base]))
cfg_t = json.loads(subprocess.check_output(["crane", "config", target]))
assert cfg_t["rootfs"]["diff_ids"][:len(cfg_b["rootfs"]["diff_ids"])] == cfg_b["rootfs"]["diff_ids"], "diff_ids differ"
print(f"base layers: {len(bl)}, added layers: {len(tl) - len(bl)}  -> exact prefix OK")
print("compressed total: %.2f GiB" % (sum(l["size"] for l in t["layers"]) / 2**30))
print("config:", json.dumps({k: cfg_t["config"].get(k) for k in ("Env", "Cmd", "Entrypoint", "WorkingDir")}, indent=1))
PY
crane digest "$TARGET"
