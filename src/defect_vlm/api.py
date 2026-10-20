"""FastAPI service: POST a photo + its category, get the inspection as JSON.

Run on a GPU machine:
    ADAPTER_DIR=/path/to/qwen2vl-mvtec-lora QUANT=4bit uvicorn defect_vlm.api:app
"""
from __future__ import annotations

import io
import os
import time
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from defect_vlm.inference import DEFECT_TYPES

app = FastAPI(title="VLM defect inspector",
              description="Fine-tuned Qwen2-VL-2B on MVTec AD. Returns defective/type/location plus a score.")

_inspector = None


def get_inspector():
    """Load the model once, on first request. Tests replace this with a fake."""
    global _inspector
    if _inspector is None:
        from defect_vlm.inference import Inspector
        _inspector = Inspector(os.environ["ADAPTER_DIR"], mode=os.environ.get("QUANT", "4bit"))
    return _inspector


class Inspection(BaseModel):
    category: str
    defective: Optional[bool]
    defect_type: Optional[str]
    location: Optional[str]
    score: float
    valid_json: bool
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/categories")
def categories():
    return DEFECT_TYPES


@app.post("/inspect", response_model=Inspection)
async def inspect(category: str = Form(...), image: UploadFile = File(...),
                  inspector=Depends(get_inspector)):
    if category not in DEFECT_TYPES:
        raise HTTPException(400, f"Unknown category '{category}'. See GET /categories.")
    try:
        img = Image.open(io.BytesIO(await image.read()))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "The uploaded file is not a readable image.")
    t0 = time.perf_counter()
    result = inspector.inspect(img, category)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return Inspection(category=category, latency_ms=latency_ms,
                      **{k: result[k] for k in ("defective", "defect_type", "location", "score", "valid_json")})
