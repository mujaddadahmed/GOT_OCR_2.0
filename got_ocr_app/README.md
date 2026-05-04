# GOT-OCR 2.0 — FastAPI Service

Wraps [GOT-OCR 2.0](https://huggingface.co/stepfun-ai/GOT-OCR2_0) in a
FastAPI application with a browser UI. Upload an image of handwritten code,
get the extracted text back instantly.

---

## Project layout

```
got_ocr_app/
├── main.py              # FastAPI app — routes, validation, lifespan
├── ocr_engine.py        # Model loading & inference (singleton)
├── static/
│   └── index.html       # Single-page upload UI
├── requirements.txt     # Python deps (torch installed separately)
├── nixpacks.toml        # Tells Railway how to build the app
├── railway.toml         # Railway deploy config (healthcheck, restart policy)
├── Procfile             # Fallback start command
└── README.md
```

---

## Deploy to Railway

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

### 2. Create a Railway project

1. [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
2. Select your repository — Railway picks up `nixpacks.toml` automatically

### 3. Add a persistent Volume (strongly recommended)

Without a volume the ~2 GB model re-downloads on every cold start (~2–3 min).

1. Service → **Volumes** → **Add Volume** → mount path `/app/.cache/huggingface`
2. Service → **Variables** → add `HF_HOME` = `/app/.cache/huggingface`

### 4. Open your app

Railway provides a public URL. UI is at `/`, Swagger docs at `/docs`.

---

## Run locally

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## API

`POST /ocr` — `multipart/form-data`

| Field | Values |
|---|---|
| `file` | JPEG / PNG / WEBP / BMP / TIFF, max 20 MB |
| `mode` | `ocr` (plain text) · `format` (structured/indented) |

```json
{ "text": "def foo(): ...", "mode": "ocr", "time_seconds": 4.2, "filename": "code.jpg" }
```

---

## Notes

- **Railway = CPU inference** — expect 30–90 s/image. For GPU speed, use RunPod or Lambda Labs.
- Model loads once at startup, shared across all requests.
