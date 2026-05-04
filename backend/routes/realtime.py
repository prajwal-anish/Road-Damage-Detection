"""
Real-Time Detection Router
POST /predict-frame — optimised for repeated, high-frequency frame inference.

Design decisions vs /predict:
  • No annotated image returned  → browser draws boxes itself (saves bandwidth)
  • No per-request logging        → avoids I/O bottleneck at 3–5 fps
  • Smaller JSON payload          → only what the frontend needs
  • Frame is resized to 640 px    → consistent inference speed
  • Non-blocking reads            → async-compatible
"""

import os
import time
import uuid
import asyncio
import torch
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────
# Configuration  (mirrors main.py — read from same env vars)
# ─────────────────────────────────────────────────────────────
_MODEL_PATH       = os.getenv("MODEL_PATH", "../model/best.pt")
_CONF_THRESHOLD   = float(os.getenv("CONF_THRESHOLD", "0.25"))
_IOU_THRESHOLD    = float(os.getenv("IOU_THRESHOLD", "0.45"))
_MAX_FRAME_SIZE_B = 5 * 1024 * 1024   # 5 MB hard limit per frame

CLASS_NAMES  = {0: "pothole", 1: "crack",   2: "manhole"}
CLASS_COLORS = {0: "#E74C3C", 1: "#F39C12", 2: "#3498DB"}
CLASS_EMOJIS = {0: "🕳️",     1: "💥",       2: "🔵"}

# Severity weights (same formula as utils/severity.py — no import to avoid circular deps)
_CLASS_WEIGHT = {0: 1.0, 1: 0.75, 2: 0.40}

def _severity(confidence: float, area_ratio: float, class_id: int) -> tuple[str, float]:
    norm_area  = min(area_ratio / 0.05, 1.0)
    raw        = 0.5 * confidence + 0.5 * norm_area
    score      = raw * _CLASS_WEIGHT.get(class_id, 0.5)
    if class_id == 0 and confidence > 0.7:
        score = max(score, 0.30)
    score = round(min(max(score, 0.0), 1.0), 4)
    if score < 0.25:   return "LOW",      score
    if score < 0.50:   return "MEDIUM",   score
    if score < 0.75:   return "HIGH",     score
    return              "CRITICAL", score

# ─────────────────────────────────────────────────────────────
# Shared model handle
# Stored on this module so main.py's startup pre-load is reused:
#   from routes.realtime import set_shared_model
#   set_shared_model(model_instance)
# ─────────────────────────────────────────────────────────────
_shared_model = None

def set_shared_model(model) -> None:
    """Called by main.py startup to hand over the already-loaded model."""
    global _shared_model
    _shared_model = model

def _get_model():
    global _shared_model
    if _shared_model is not None:
        return _shared_model
    # Fallback: lazy-load if main.py hasn't set it yet
    from ultralytics import YOLO
    from pathlib import Path
    if not Path(_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found at {_MODEL_PATH}. "
            "Place best.pt in the model/ directory and restart the server."
        )
    
    # PyTorch 2.6+ requires weights_only=False for models with complex serialization
    original_load = torch.load
    
    def load_with_fallback(f, *args, **kwargs):
        try:
            return original_load(f, *args, weights_only=True, **kwargs)
        except Exception:
            return original_load(f, *args, weights_only=False, **kwargs)
    
    torch.load = load_with_fallback
    try:
        _shared_model = YOLO(_MODEL_PATH)
    finally:
        torch.load = original_load
    
    return _shared_model

# ─────────────────────────────────────────────────────────────
# Pydantic schemas (lightweight — no annotated image field)
# ─────────────────────────────────────────────────────────────

class FrameBBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float

class FrameDetection(BaseModel):
    id:            int
    class_id:      int
    class_name:    str
    class_emoji:   str
    confidence:    float = Field(..., ge=0.0, le=1.0)
    bbox:          FrameBBox
    severity:      str
    severity_score: float
    color:         str

class FrameResponse(BaseModel):
    frame_id:       str
    frame_width:    int
    frame_height:   int
    detections:     list[FrameDetection]
    total:          int
    inference_ms:   float
    conf_threshold: float

# ─────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────
router = APIRouter()


@router.post(
    "/predict-frame",
    response_model=FrameResponse,
    summary="Real-time frame inference",
    description=(
        "Optimised for repeated calls at 2–5 fps from a browser webcam or video stream. "
        "Send a JPEG/PNG frame as multipart; receive bounding-box JSON. "
        "No annotated image is returned — the browser draws boxes on a canvas overlay."
    ),
)
async def predict_frame(
    frame: UploadFile = File(..., description="JPEG/PNG frame captured by the browser"),
    conf:  float      = Query(_CONF_THRESHOLD, ge=0.01, le=0.99, description="Confidence threshold"),
    iou:   float      = Query(_IOU_THRESHOLD,  ge=0.01, le=0.99, description="IoU (NMS) threshold"),
):
    frame_id = str(uuid.uuid4())[:8]

    # ── Size guard ──────────────────────────────────────────
    raw = await frame.read()
    if len(raw) > _MAX_FRAME_SIZE_B:
        raise HTTPException(
            status_code=413,
            detail=f"Frame too large ({len(raw)//1024} KB). Max 5 MB.",
        )

    # ── Decode ──────────────────────────────────────────────
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode frame. Send a valid JPEG/PNG.")

    h, w = img.shape[:2]
    img_area = float(h * w) or 1.0

    # ── Model ───────────────────────────────────────────────
    try:
        model = _get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # ── Inference  (run in thread-pool — YOLO is sync) ──────
    t0 = time.perf_counter()
    results = await asyncio.get_event_loop().run_in_executor(
        None,                           # default ThreadPoolExecutor
        lambda: model.predict(
            img,
            conf=conf,
            iou=iou,
            imgsz=640,
            verbose=False,
            augment=False,
        )
    )
    inference_ms = (time.perf_counter() - t0) * 1000.0

    # ── Parse detections ────────────────────────────────────
    detections: list[FrameDetection] = []
    result = results[0]

    for det_id, box in enumerate(result.boxes):
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().numpy())
        confidence = float(box.conf[0])
        class_id   = int(box.cls[0])
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

        bw         = x2 - x1
        bh         = y2 - y1
        area_ratio = (bw * bh) / img_area
        severity, sev_score = _severity(confidence, area_ratio, class_id)

        detections.append(FrameDetection(
            id            = det_id,
            class_id      = class_id,
            class_name    = class_name,
            class_emoji   = CLASS_EMOJIS.get(class_id, "❓"),
            confidence    = round(confidence, 4),
            bbox          = FrameBBox(
                x1     = round(x1, 2),
                y1     = round(y1, 2),
                x2     = round(x2, 2),
                y2     = round(y2, 2),
                width  = round(bw, 2),
                height = round(bh, 2),
            ),
            severity       = severity,
            severity_score = sev_score,
            color          = CLASS_COLORS.get(class_id, "#FFFFFF"),
        ))

    return FrameResponse(
        frame_id       = frame_id,
        frame_width    = w,
        frame_height   = h,
        detections     = detections,
        total          = len(detections),
        inference_ms   = round(inference_ms, 2),
        conf_threshold = conf,
    )
