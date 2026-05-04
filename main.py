"""
main.py — GOT-OCR 2.0 FastAPI service
"""

import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ocr_engine import model_status, run_ocr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

ALLOWED_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff",
}
MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB

# No lifespan hook — model loads lazily on first /ocr request
app = FastAPI(
    title="GOT-OCR 2.0 API",
    description="Handwritten code recognition powered by GOT-OCR 2.0",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    """Liveness probe — always returns 200 as soon as the server is up."""
    return {"status": "ok"}


@app.get("/status")
async def status():
    """Shows whether the model is loaded, loading, or errored."""
    return model_status()


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("ocr"),
):
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
