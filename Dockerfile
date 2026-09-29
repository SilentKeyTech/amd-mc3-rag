# Mini-Challenge 3 (RAG) submission.
# Mandated base, checked by layer identity. Do NOT squash or flatten this image.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

# ---- Python deps, without letting pip touch the ROCm torch (trap 2) --------
# The constraints file pins every torch-family package to the exact build that
# ships in the base image. If anything wants a different torch, the build
# FAILS here instead of silently installing a CUDA wheel over ROCm.
COPY requirements.txt /tmp/requirements.txt
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|pytorch-triton-rocm|numpy|pillow)==' > /tmp/constraints.txt \
 && cat /tmp/constraints.txt \
 && pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/requirements.txt \
 && python3 -c "import torch, torchvision, sys; v = torch.__version__; print('torch', v, 'torchvision', torchvision.__version__); sys.exit(0 if '+rocm' in v and '+rocm' in torchvision.__version__ else 'torch is no longer the ROCm build: ' + v)"

# ---- Weights ship in the image: no network at evaluation --------------------
ARG MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}', local_dir='/models/vlm', allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model','*.tiktoken'])" \
 && rm -rf /opt/hf/hub \
 && du -sh /models/vlm && ls /models/vlm

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MC3_MODEL=/models/vlm \
    MIOPEN_USER_DB_PATH=/tmp/miopen \
    TOKENIZERS_PARALLELISM=false

COPY app/ /app/
RUN mkdir -p /app/corpus /app/output /app/index /app/logs \
 && python3 -c "import sys; sys.path.insert(0, '/app'); import rag.pipeline, rag.llm, transformers, pypdf; print('imports ok, transformers', transformers.__version__)"

# The resident server loads the model once and holds the index; app.py is a
# thin client (trap 1). The loop restarts the server if it ever dies, and keeps
# the container running for the whole evaluation.
CMD ["sh", "-c", "export MC3_CONTAINER_START=$(date +%s); while true; do python3 /app/server.py; echo 'server exited, restarting' >&2; sleep 1; done"]
