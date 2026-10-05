"""
SurfShield API Server
Run: uvicorn main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dl_model import load_dl_model, predict_dl
from fusion import fuse
from ml_model import load_ml_model, predict_ml


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Server] Loading models...")
    load_dl_model()
    load_ml_model()
    print("[Server] Both models ready. Listening on http://127.0.0.1:8000")
    yield


app = FastAPI(title="SurfShield", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Chrome extension origin
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class URLRequest(BaseModel):
    url: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: URLRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is empty")

    p_dl = predict_dl(url)
    p_ml = predict_ml(url)

    result         = fuse(p_dl, p_ml)
    result["url"]  = url
    return result
