"""
ocr_engine.py
─────────────
Singleton loader for GOT-OCR 2.0.
Import `run_ocr` from here — it is safe to call from multiple requests
because the model is loaded once at startup and reused.
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_ID = "stepfun-ai/GOT-OCR2_0"

# ── Module-level singletons ────────────────────────────────────────────────────
_tokenizer = None
_model = None


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # GOT-OCR was trained for GPU; CPU works but is very slow
    logger.warning("CUDA not available — running on CPU. Expect slow inference.")
    return "cpu"


def load_model() -> None:
    """Load tokenizer + model into module-level singletons.

    Call once at application startup (FastAPI lifespan).
    Thread-safe for read-only inference after loading.
    """
    global _tokenizer, _model

    if _model is not None:
        return  # already loaded

    device = _get_device()
    logger.info("Loading GOT-OCR 2.0 tokenizer from %s …", MODEL_ID)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    logger.info("Loading GOT-OCR 2.0 model (first run downloads ~2 GB) …")
    load_kwargs = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        pad_token_id=_tokenizer.eos_token_id,
    )
    if device == "cuda":
        load_kwargs["device_map"] = "cuda"
        _model = AutoModel.from_pretrained(MODEL_ID, **load_kwargs).eval().cuda()
    else:
        _model = AutoModel.from_pretrained(MODEL_ID, **load_kwargs).eval()

    logger.info("GOT-OCR 2.0 ready on %s", device)


def run_ocr(
    image_bytes: bytes,
    mode: Literal["ocr", "format"] = "ocr",
    filename: str = "upload.jpg",
) -> dict:
    """
    Run GOT-OCR 2.0 on raw image bytes.

    Parameters
    ----------
    image_bytes : raw bytes of the uploaded image file
    mode        : 'ocr'    → plain text output
                  'format' → structured / formatted output (LaTeX-style)
    filename    : original filename (used only to choose a tmp suffix)

    Returns
    -------
    dict with keys:
        text          str   — recognised text
        mode          str   — mode used
        time_seconds  float — wall-clock inference time
        error         str | None
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")

    suffix = Path(filename).suffix or ".jpg"

    # Write bytes to a real temp file because model.chat() needs a file path
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    start = time.perf_counter()
    try:
        result = _model.chat(_tokenizer, tmp_path, ocr_type=mode)
        elapsed = time.perf_counter() - start
        return {
            "text": result.strip() if result else "",
            "mode": mode,
            "time_seconds": round(elapsed, 3),
            "error": None,
        }
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        return {
            "text": "",
            "mode": mode,
            "time_seconds": round(time.perf_counter() - start, 3),
            "error": str(exc),
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)
