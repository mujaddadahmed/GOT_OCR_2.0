"""
main.py — GOT-OCR 2.0 FastAPI service
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ocr_engine import model_status, run_ocr, start_background_load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
MAX_FILE_BYTES = 20 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start downloading/loading the model immediately in the background.
    # The server becomes healthy right away; /ocr returns 503 until ready.
    logger.info("Starting background model load …")
    start_background_load()
    yield


app = FastAPI(
    title="GOT-OCR 2.0 API",
    description="Handwritten code recognition powered by GOT-OCR 2.0",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    """Liveness probe — always 200 once the server is up."""
    return {"status": "ok"}


@app.get("/status")
async def status():
    """Readiness probe — poll this to know when the model is ready."""
    return model_status()


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("ocr"),
):
    # Check model readiness first
    s = model_status()
    if s["loading"]:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Please wait a moment and try again.",
        )
    if s["error"]:
        raise HTTPException(status_code=500, detail=f"Model load failed: {s['error']}")

    if (file.content_type or "") not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type '{file.content_type}'. "
                   f"Accepted: {', '.join(sorted(ALLOWED_TYPES))}",
        )
    if mode not in ("ocr", "format"):
        raise HTTPException(status_code=422, detail="mode must be 'ocr' or 'format'")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 20 MB limit.")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info("OCR request — file=%s size=%d mode=%s", file.filename, len(image_bytes), mode)

    result = run_ocr(image_bytes, mode=mode, filename=file.filename or "upload.jpg")
    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])

    return JSONResponse(content={
        "text": result["text"],
        "mode": result["mode"],
        "time_seconds": result["time_seconds"],
        "filename": file.filename,
    })
