"""
ocr_engine.py
─────────────
Lazy singleton loader for GOT-OCR 2.0.
The model is downloaded and loaded on the FIRST call to run_ocr(),
not at startup — so the server becomes healthy immediately.
"""

import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_ID = "stepfun-ai/GOT-OCR2_0"

_tokenizer = None
_model = None
_loading = False
_load_error: str | None = None
_lock = threading.Lock()


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    logger.warning("CUDA not available — running on CPU. Inference will be slow.")
    return "cpu"


def _load_model_internal() -> None:
    global _tokenizer, _model, _loading, _load_error

    device = _get_device()
    logger.info("Loading GOT-OCR 2.0 tokenizer …")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    logger.info("Loading GOT-OCR 2.0 model (may download ~2 GB on first run) …")
    load_kwargs = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        pad_token_id=tok.eos_token_id,
    )
    if device == "cuda":
        load_kwargs["device_map"] = "cuda"
        mdl = AutoModel.from_pretrained(MODEL_ID, **load_kwargs).eval().cuda()
    else:
        mdl = AutoModel.from_pretrained(MODEL_ID, **load_kwargs).eval()

    _tokenizer = tok
    _model = mdl
    logger.info("GOT-OCR 2.0 ready on %s", device)


def ensure_loaded() -> None:
    """Load the model if not already loaded. Thread-safe. Raises on failure."""
    global _loading, _load_error

    if _model is not None:
        return
    if _load_error:
        raise RuntimeError(f"Model failed to load previously: {_load_error}")

    with _lock:
        # Double-checked locking
        if _model is not None:
            return
        if _load_error:
            raise RuntimeError(f"Model failed to load previously: {_load_error}")

        _loading = True
        try:
            _load_model_internal()
        except Exception as exc:
            _load_error = str(exc)
            logger.exception("Model load failed: %s", exc)
            raise
        finally:
            _loading = False


def model_status() -> dict:
    return {
        "loaded": _model is not None,
        "loading": _loading,
        "error": _load_error,
    }


def run_ocr(
    image_bytes: bytes,
    mode: Literal["ocr", "format"] = "ocr",
    filename: str = "upload.jpg",
) -> dict:
    ensure_loaded()

    suffix = Path(filename).suffix or ".jpg"
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
