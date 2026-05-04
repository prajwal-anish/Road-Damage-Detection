"""
Road Surface Damage Detection System — FastAPI Backend
Handles image upload, YOLO inference, severity scoring, and logging.
"""

import os
import io
import time
import uuid
import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
from ultralytics import YOLO

from utils.severity import compute_severity, SeverityLevel
from utils.logger import setup_logger, log_detection_event
from routes.realtime import router as realtime_router, set_shared_model

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "../model/best.pt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
MAX_IMAGE_SIZE_MB = float(os.getenv("MAX_IMAGE_SIZE_MB", "10.0"))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = {0: "pothole", 1: "crack", 2: "manhole"}
CLASS_COLORS = {0: "#E74C3C", 1: "#2ECC71", 2: "#3498DB"}
CLASS_EMOJIS = {0: "🕳️", 1: "💥", 2: "🔵"}

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────────────────────
logger = setup_logger("road_damage_api", LOGS_DIR / "api.log")

# ──────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Road Surface Damage Detection API",
    description="Detect potholes, cracks, and manholes using YOLOv8",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──────────────────────────────────────────
app.include_router(realtime_router, tags=["Real-Time Detection"])

# ──────────────────────────────────────────────────────────────
# Global model state
# ──────────────────────────────────────────────────────────────
_model: Optional[YOLO] = None
_model_load_time: Optional[float] = None
_inference_count: int = 0
_total_inference_time: float = 0.0


def load_model() -> YOLO:
    global _model, _model_load_time
    if _model is None:
        if not Path(MODEL_PATH).exists():
            logger.error(f"Model not found at: {MODEL_PATH}")
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                "Please place best.pt in the model/ directory."
            )
        logger.info(f"Loading YOLO model from {MODEL_PATH}...")
        t0 = time.time()
        
        # PyTorch 2.6+ requires weights_only=False for models with complex serialization
        # Temporarily patch torch.load to disable weights_only check
        import torch.serialization
        original_load = torch.load
        
        def load_with_fallback(f, *args, **kwargs):
            try:
                # Try with weights_only=True first (secure mode)
                return original_load(f, *args, weights_only=True, **kwargs)
            except Exception:
                # Fall back to weights_only=False for this trusted model
                logger.warning("Model requires weights_only=False due to complex serialization")
                return original_load(f, *args, weights_only=False, **kwargs)
        
        torch.load = load_with_fallback
        try:
            _model = YOLO(MODEL_PATH)
        finally:
            torch.load = original_load
        
        _model_load_time = time.time() - t0
        logger.info(f"Model loaded in {_model_load_time:.2f}s")
    return _model


@app.on_event("startup")
async def startup_event():
    """Pre-load model on startup to avoid cold start delay."""
    try:
        m = load_model()
        set_shared_model(m)          # hand model to realtime router
        logger.info("✅ API startup complete — model ready")
    except FileNotFoundError as e:
        logger.warning(f"⚠️  Model not loaded on startup: {e}")


# ──────────────────────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────────────────────
class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    center_x: float
    center_y: float


class Detection(BaseModel):
    id: int
    class_id: int
    class_name: str
    class_emoji: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    area_px: float
    area_ratio: float = Field(..., description="Fraction of image area")
    severity: str
    severity_score: float = Field(..., ge=0.0, le=1.0)
    color: str


class DetectionSummary(BaseModel):
    total_detections: int
    class_counts: dict
    overall_severity: str
    overall_severity_score: float
    dominant_class: Optional[str]
    affected_area_ratio: float


class PredictionResponse(BaseModel):
    request_id: str
    timestamp: str
    image_width: int
    image_height: int
    detections: list[Detection]
    summary: DetectionSummary
    inference_time_ms: float
    model_confidence_threshold: float
    annotated_image_base64: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    total_requests: int
    avg_inference_ms: float
    uptime_seconds: float


class StatsResponse(BaseModel):
    total_inferences: int
    avg_inference_ms: float
    class_detections: dict


# ──────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────
def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes to numpy array (BGR)."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Ensure it is a valid image file.")
    return img


def encode_image_base64(img: np.ndarray) -> str:
    """Encode numpy array (BGR) to base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode("utf-8")


def draw_annotated_image(
    img: np.ndarray,
    detections: list[Detection],
    conf_threshold: float
) -> np.ndarray:
    """Draw bounding boxes and labels on image."""
    annotated = img.copy()
    h, w = annotated.shape[:2]

    color_map = {
        "pothole": (60, 76, 231),   # BGR: #E74C3C
        "crack": (113, 204, 46),    # BGR: #2ECC71
        "manhole": (219, 152, 52),  # BGR: #3498DB
    }

    for det in detections:
        x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
        x2, y2 = int(det.bbox.x2), int(det.bbox.y2)
        color = color_map.get(det.class_name, (255, 255, 255))

        # Box thickness based on severity
        thickness = {"LOW": 2, "MEDIUM": 3, "HIGH": 4, "CRITICAL": 5}.get(det.severity, 2)

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Draw corner accents
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        for cx, cy, dx, dy in [
            (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(annotated, (cx, cy), (cx + dx * corner_len, cy), color, thickness + 1)
            cv2.line(annotated, (cx, cy), (cx, cy + dy * corner_len), color, thickness + 1)

        # Label background
        label = f"{det.class_name.upper()} {det.confidence:.2f} [{det.severity}]"
        font_scale = 0.5
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness
        )
        label_y = max(y1 - 8, text_h + 8)
        cv2.rectangle(annotated, (x1, label_y - text_h - 4), 
                      (x1 + text_w + 4, label_y + baseline), color, -1)
        cv2.putText(annotated, label, (x1 + 2, label_y - 2),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), font_thickness)

    # Overlay stats
    stats_text = [
        f"Detections: {len(detections)}",
        f"Threshold: {conf_threshold:.2f}",
    ]
    for i, txt in enumerate(stats_text):
        y_pos = 25 + i * 22
        cv2.putText(annotated, txt, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3)
        cv2.putText(annotated, txt, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)

    return annotated


_startup_time = time.time()
_class_detection_counts = {name: 0 for name in CLASS_NAMES.values()}


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "🚗 Road Surface Damage Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "stats": "GET /stats",
            "docs": "GET /docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Returns API health status and model info."""
    avg_ms = (_total_inference_time / _inference_count * 1000) if _inference_count > 0 else 0
    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_path=MODEL_PATH,
        total_requests=_inference_count,
        avg_inference_ms=round(avg_ms, 2),
        uptime_seconds=round(time.time() - _startup_time, 1),
    )


@app.get("/stats", response_model=StatsResponse, tags=["Monitoring"])
async def get_stats():
    """Returns cumulative inference statistics."""
    avg_ms = (_total_inference_time / _inference_count * 1000) if _inference_count > 0 else 0
    return StatsResponse(
        total_inferences=_inference_count,
        avg_inference_ms=round(avg_ms, 2),
        class_detections=_class_detection_counts,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Detection"])
async def predict(
    file: UploadFile = File(..., description="Road image (JPG, PNG, BMP, WEBP)"),
    conf: float = Query(CONF_THRESHOLD, ge=0.01, le=0.99, description="Confidence threshold"),
    iou: float = Query(IOU_THRESHOLD, ge=0.01, le=0.99, description="IoU threshold"),
    include_annotated_image: bool = Query(True, description="Include annotated image in response"),
):
    """
    Run road damage detection on the uploaded image.

    Returns detected potholes, cracks, and manholes with:
    - Bounding boxes
    - Confidence scores
    - Severity assessment
    - Annotated image (base64)
    """
    global _inference_count, _total_inference_time

    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Prediction request — file={file.filename}, conf={conf}")

    # ── Validate file ──
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large. Max size: {MAX_IMAGE_SIZE_MB}MB",
        )

    # ── Decode image ──
    try:
        img_bgr = decode_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    h, w = img_bgr.shape[:2]
    img_area = h * w

    # ── Load model ──
    try:
        model = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # ── Inference ──
    t0 = time.time()
    results = model.predict(
        img_bgr,
        conf=conf,
        iou=iou,
        imgsz=640,
        verbose=False,
        augment=False,
    )
    inference_time = time.time() - t0

    _inference_count += 1
    _total_inference_time += inference_time

    # ── Parse detections ──
    detections: list[Detection] = []
    result = results[0]

    for det_id, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

        box_w = x2 - x1
        box_h = y2 - y1
        area_px = box_w * box_h
        area_ratio = area_px / img_area

        severity, severity_score = compute_severity(confidence, area_ratio, class_id)

        det = Detection(
            id=det_id,
            class_id=class_id,
            class_name=class_name,
            class_emoji=CLASS_EMOJIS.get(class_id, "❓"),
            confidence=round(confidence, 4),
            bbox=BoundingBox(
                x1=round(float(x1), 2),
                y1=round(float(y1), 2),
                x2=round(float(x2), 2),
                y2=round(float(y2), 2),
                width=round(float(box_w), 2),
                height=round(float(box_h), 2),
                center_x=round(float((x1 + x2) / 2), 2),
                center_y=round(float((y1 + y2) / 2), 2),
            ),
            area_px=round(area_px, 2),
            area_ratio=round(area_ratio, 6),
            severity=severity,
            severity_score=round(severity_score, 4),
            color=CLASS_COLORS.get(class_id, "#FFFFFF"),
        )
        detections.append(det)
        _class_detection_counts[class_name] = _class_detection_counts.get(class_name, 0) + 1

    # ── Summary ──
    class_counts = {}
    for det in detections:
        class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

    overall_score = max((d.severity_score for d in detections), default=0.0)
    overall_severity = SeverityLevel.from_score(overall_score).label

    dominant_class = (
        max(class_counts, key=class_counts.get) if class_counts else None
    )
    affected_area = sum(d.area_ratio for d in detections)

    summary = DetectionSummary(
        total_detections=len(detections),
        class_counts=class_counts,
        overall_severity=overall_severity,
        overall_severity_score=round(overall_score, 4),
        dominant_class=dominant_class,
        affected_area_ratio=round(min(affected_area, 1.0), 6),
    )

    # ── Annotated image ──
    annotated_b64 = None
    if include_annotated_image:
        annotated_img = draw_annotated_image(img_bgr, detections, conf)
        annotated_b64 = encode_image_base64(annotated_img)

    # ── Log event ──
    log_detection_event(
        logger=logger,
        request_id=request_id,
        filename=file.filename,
        num_detections=len(detections),
        class_counts=class_counts,
        inference_ms=inference_time * 1000,
        severity=overall_severity,
    )

    response = PredictionResponse(
        request_id=request_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        image_width=w,
        image_height=h,
        detections=detections,
        summary=summary,
        inference_time_ms=round(inference_time * 1000, 2),
        model_confidence_threshold=conf,
        annotated_image_base64=annotated_b64,
    )

    logger.info(
        f"[{request_id}] ✅ Done — {len(detections)} detections in {inference_time*1000:.1f}ms"
    )
    return response


@app.post("/predict/batch", tags=["Detection"])
async def predict_batch(
    files: list[UploadFile] = File(...),
    conf: float = Query(CONF_THRESHOLD, ge=0.01, le=0.99),
):
    """Run detection on multiple images (max 10)."""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Max 10 images per batch request")

    results_list = []
    for f in files:
        try:
            response = await predict(
                file=f, conf=conf, iou=IOU_THRESHOLD, include_annotated_image=False
            )
            results_list.append({"filename": f.filename, "result": response.dict()})
        except HTTPException as e:
            results_list.append({"filename": f.filename, "error": e.detail})

    return {"batch_size": len(files), "results": results_list}
