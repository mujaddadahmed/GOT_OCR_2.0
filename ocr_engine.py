"""
ocr_engine.py — GOT-OCR 2.0 singleton with background loading
"""

import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

MODEL_ID = "stepfun-ai/GOT-OCR2_0"

_tokenizer = None
_model = None
_load_error: str | None = None
_ready = threading.Event()   # set once model is fully loaded


def _load():
    global _tokenizer, _model, _load_error
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.warning("CUDA not available — running on CPU. Inference will be slow.")

        logger.info("Loading GOT-OCR 2.0 tokenizer …")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        logger.info("Loading GOT-OCR 2.0 model (may download ~2 GB on first run) …")
        kwargs = dict(
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            pad_token_id=_tokenizer.eos_token_id,
        )
        if device == "cuda":
            kwargs["device_map"] = "cuda"
            _model = AutoModel.from_pretrained(MODEL_ID, **kwargs).eval().cuda()
        else:
            _model = AutoModel.from_pretrained(MODEL_ID, **kwargs).eval()

        logger.info("GOT-OCR 2.0 ready on %s", device)

    except Exception as exc:
        _load_error = str(exc)
        logger.exception("Model load failed: %s", exc)
    finally:
        _ready.set()   # always unblock waiters, even on failure


def start_background_load():
    """Kick off model loading in a daemon thread. Call once at app startup."""
    t = threading.Thread(target=_load, name="model-loader", daemon=True)
    t.start()


def wait_until_ready(timeout: float = 5.0) -> bool:
    """Block up to `timeout` seconds. Returns True if model is loaded."""
    return _ready.wait(timeout=timeout)


def model_status() -> dict:
    ready = _ready.is_set()
    return {
        "loaded": ready and _load_error is None,
        "loading": not ready,
        "error": _load_error,
    }


def run_ocr(
    image_bytes: bytes,
    mode: Literal["ocr", "format"] = "ocr",
    filename: str = "upload.jpg",
) -> dict:
    if not _ready.is_set():
        raise RuntimeError("Model is still loading.")
    if _load_error:
        raise RuntimeError(f"Model failed to load: {_load_error}")

    suffix = Path(filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    start = time.perf_counter()
    try:
        result = _model.chat(_tokenizer, tmp_path, ocr_type=mode)
        return {
            "text": result.strip() if result else "",
            "mode": mode,
            "time_seconds": round(time.perf_counter() - start, 3),
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
