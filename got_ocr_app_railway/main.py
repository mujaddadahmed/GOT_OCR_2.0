"""
main.py — GOT-OCR 2.0 FastAPI service
─────────────────────────────────────
Endpoints
  GET  /           → serves the HTML UI
  POST /ocr        → accepts an image upload, returns JSON with recognised text
  GET  /health     → liveness / readiness probe
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ocr_engine import load_model, run_ocr

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# ── Allowed MIME types ─────────────────────────────────────────────────────────
ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB hard limit


# ── Lifespan: load model once at startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Application startup: loading GOT-OCR 2.0 ===")
    load_model()
    logger.info("=== Model ready — accepting requests ===")
    yield
    logger.info("=== Application shutdown ===")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GOT-OCR 2.0 API",
    description="Handwritten code recognition powered by GOT-OCR 2.0",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the single-page HTML interface."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    """Liveness / readiness probe."""
    return {"status": "ok"}


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP, BMP, TIFF)"),
    mode: str = Form("ocr", description="'ocr' for plain text, 'format' for structured output"),
):
    """
    Accept an uploaded image and return the recognised text from GOT-OCR 2.0.

    - **file**: image to recognise
    - **mode**: `ocr` (plain text) or `format` (structured / formatted)
    """
    # ── Validate content type ──────────────────────────────────────────────────
    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. "
                   f"Accepted: {', '.join(sorted(ALLOWED_TYPES))}",
        )

    # ── Validate mode ──────────────────────────────────────────────────────────
    if mode not in ("ocr", "format"):
        raise HTTPException(
            status_code=422,
            detail="mode must be 'ocr' or 'format'",
        )

    # ── Read file with size guard ──────────────────────────────────────────────
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_BYTES // (1024*1024)} MB.",
        )
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info(
        "OCR request — file=%s size=%d bytes mode=%s",
        file.filename,
        len(image_bytes),
        mode,
    )

    # ── Run inference ──────────────────────────────────────────────────────────
    result = run_ocr(image_bytes, mode=mode, filename=file.filename or "upload.jpg")

    if result["error"]:
        logger.error("Inference error: %s", result["error"])
        raise HTTPException(status_code=500, detail=result["error"])

    return JSONResponse(
        content={
            "text": result["text"],
            "mode": result["mode"],
            "time_seconds": result["time_seconds"],
            "filename": file.filename,
        }
    )
