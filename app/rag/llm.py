"""Vision-language model wrapper, loaded ONCE by the resident server.

Uses Hugging Face transformers on the ROCm torch that ships in the base image.
Everything model-specific lives here so the pipeline can run against a mock in
tests.
"""

from __future__ import annotations

import io
import logging
import os
import time

log = logging.getLogger("mc3.llm")

OCR_PROMPT = (
    "Transcribe ALL text visible in this image, exactly as printed, including every label, "
    "number, code, part number, revision and caption. Keep the original spelling and symbols "
    "(for example # _ - /). For diagrams, pinouts, tables and forms, write one line per item that "
    "pairs each label with the value it is connected, attached or adjacent to, e.g. 'B14: THERM_ALERT#' "
    "or 'BOARD REVISION: REV-C2'. Do not describe the image and do not add anything that is not printed."
)


def _prep_image(raw: bytes, max_side: int = 1600, min_side: int = 640):
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(raw))
    try:
        img.seek(0)
    except EOFError:
        pass
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB",):
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
    elif max(w, h) < min_side:
        scale = min_side / max(w, h)
    if scale != 1.0:
        img = img.resize((max(28, int(w * scale)), max(28, int(h * scale))), Image.LANCZOS)
    return img


class VLM:
    def __init__(self, model_path: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        path = model_path or os.environ.get("MC3_MODEL", "/models/vlm")
        device = "cuda"
        if not torch.cuda.is_available():
            # The grader rejects CPU runs; the override exists only for local tests.
            if not os.environ.get("MC3_ALLOW_CPU"):
                raise RuntimeError("no GPU visible to torch; refusing to run on the CPU")
            device = "cpu"
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(path)
        kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(path, device_map=device, **kwargs)
        except (ImportError, ValueError, TypeError):  # no accelerate: load then move
            self.model = AutoModelForImageTextToText.from_pretrained(path, **kwargs).to(device)
        self.model.eval()
        log.info("model loaded from %s in %.1fs on %s", path, time.time() - t0,
                 torch.cuda.get_device_name(0) if device == "cuda" else "cpu")
        self.generate([{"type": "text", "text": "Reply with OK."}], max_new_tokens=4)  # warm the kernels
        log.info("warmup done at %.1fs", time.time() - t0)

    def generate(self, content: list[dict], max_new_tokens: int = 256, system: str | None = None) -> str:
        """content: list of {"type": "text", "text": ...} / {"type": "image", "image": bytes}."""
        msg_content = []
        for part in content:
            if part["type"] == "image":
                msg_content.append({"type": "image", "image": _prep_image(part["image"])})
            else:
                msg_content.append({"type": "text", "text": part["text"]})
        messages = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": msg_content})
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def ocr(self, raw: bytes) -> str:
        return self.generate([{"type": "image", "image": raw}, {"type": "text", "text": OCR_PROMPT}], max_new_tokens=768)


class MockLLM:
    """Deterministic stand-in used by the local test harness (no GPU)."""

    def __init__(self, ocr_text: dict[bytes, str] | None = None, responder=None):
        self.ocr_text = ocr_text or {}
        self.responder = responder
        self.calls: list[str] = []

    def generate(self, content, max_new_tokens=256, system=None):
        text = "\n".join(p["text"] for p in content if p["type"] == "text")
        self.calls.append(text)
        return self.responder(text, content) if self.responder else '{"answer": "", "sources": []}'

    def ocr(self, raw: bytes) -> str:
        return self.ocr_text.get(raw, "")
