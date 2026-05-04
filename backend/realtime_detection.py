"""
Real-Time Road Damage Detection via Webcam
Uses OpenCV + YOLOv8 for live inference.

Usage:
    python realtime_detection.py
    python realtime_detection.py --source 0          # webcam
    python realtime_detection.py --source video.mp4  # video file
    python realtime_detection.py --conf 0.3 --source 0
"""

import argparse
import time
import collections
import pickle
import functools
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
# PyTorch 2.6+ Compatibility: Monkey-patch torch.load for safe deserialization
# ──────────────────────────────────────────────────────────────
_original_torch_load = torch.load

@functools.wraps(_original_torch_load)
def _patched_torch_load(f, map_location=None, **kwargs):
    """Wrap torch.load to fall back to weights_only=False on pickling errors."""
    try:
        return _original_torch_load(f, map_location=map_location, **kwargs)
    except (pickle.UnpicklingError, RuntimeError) as e:
        if "weights_only" in str(e) or "not an allowed global" in str(e):
            kwargs['weights_only'] = False
            return _original_torch_load(f, map_location=map_location, **kwargs)
        raise

torch.load = _patched_torch_load

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
MODEL_PATH = "../model/best.pt"
CONF_THRESHOLD = 0.30
IOU_THRESHOLD = 0.45
TARGET_FPS = 30
IMGSZ = 640
SKIP_FRAMES = 1  # Run inference every N frames for speed

CLASS_NAMES = {0: "Pothole", 1: "Crack", 2: "Manhole"}
CLASS_COLORS_BGR = {
    0: (60, 76, 231),    # #E74C3C — Red
    1: (113, 204, 46),   # #2ECC71 — Green
    2: (219, 152, 52),   # #3498DB — Blue
}
SEVERITY_COLORS = {
    "LOW":      (46, 204, 113),   # Green
    "MEDIUM":   (243, 156, 18),   # Orange
    "HIGH":     (230, 126, 34),   # Dark Orange
    "CRITICAL": (231, 76, 60),    # Red
}


def compute_severity_score(confidence: float, area_ratio: float, class_id: int) -> str:
    """Quick severity computation for real-time use."""
    weights = {0: 1.0, 1: 0.75, 2: 0.4}
    norm_area = min(area_ratio / 0.05, 1.0)
    score = (0.5 * confidence + 0.5 * norm_area) * weights.get(class_id, 0.5)
    if score < 0.25:
        return "LOW"
    elif score < 0.50:
        return "MEDIUM"
    elif score < 0.75:
        return "HIGH"
    else:
        return "CRITICAL"


def draw_hud(frame, fps, total_dets, class_counts, conf_threshold):
    """Draw heads-up display overlay on frame."""
    h, w = frame.shape[:2]
    
    # Semi-transparent HUD background (top-right)
    hud_w, hud_h = 220, 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - hud_w - 10, 8), (w - 10, hud_h + 8), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # FPS
    fps_color = (0, 255, 0) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - hud_w, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, fps_color, 1)
    
    cv2.putText(frame, f"Conf: {conf_threshold:.2f}", (w - hud_w, 52),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.putText(frame, f"Total: {total_dets}", (w - hud_w, 74),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)
    
    # Per-class counts
    for i, (cls_id, name) in enumerate(CLASS_NAMES.items()):
        count = class_counts.get(cls_id, 0)
        color = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))
        cv2.putText(frame, f"{name}: {count}", (w - hud_w, 96 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    
    # "LIVE" indicator
    cv2.circle(frame, (20, 20), 7, (0, 0, 255), -1)
    cv2.putText(frame, "LIVE", (32, 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 255), 1)


def draw_detection(frame, box, class_id, confidence, severity):
    """Draw a single detection on frame."""
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    h, w = frame.shape[:2]
    area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)

    color = CLASS_COLORS_BGR.get(class_id, (255, 255, 255))
    sev_color = SEVERITY_COLORS.get(severity, (255, 255, 255))
    thickness = 2

    # Main bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Severity-colored corners
    corner_len = max(12, min(20, (x2 - x1) // 5))
    for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx,cy), (cx+dx*corner_len, cy), sev_color, 3)
        cv2.line(frame, (cx,cy), (cx, cy+dy*corner_len), sev_color, 3)

    # Label
    label = f"{CLASS_NAMES[class_id]} {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ly = max(y1 - 6, th + 6)
    cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), color, -1)
    cv2.putText(frame, label, (x1 + 3, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Severity badge
    sev_label = severity[:3]
    cv2.putText(frame, sev_label, (x2 - 28, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, sev_color, 1)


def run_realtime(source, conf_threshold, show_fps=True):
    """Main real-time detection loop."""
    print(f"🚀 Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded")

    cap = cv2.VideoCapture(source if isinstance(source, str) else int(source))
    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    print(f"✅ Stream opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
          f"{int(cap.get(cv2.CAP_PROP_FPS))}fps")
    print("📌 Controls: Q=quit | +/- = conf threshold | S=screenshot | P=pause")

    # FPS smoothing
    fps_buffer = collections.deque(maxlen=30)
    frame_count = 0
    prev_boxes = []
    paused = False
    total_detections_session = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("📛 Stream ended")
                break

            frame_count += 1
            t_start = time.perf_counter()

            # Run inference every SKIP_FRAMES frames
            if frame_count % (SKIP_FRAMES + 1) == 0:
                results = model.predict(
                    frame,
                    conf=conf_threshold,
                    iou=IOU_THRESHOLD,
                    imgsz=IMGSZ,
                    verbose=False,
                    stream=False,
                )
                prev_boxes = []
                if results and results[0].boxes:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0])
                        conf_val = float(box.conf[0])
                        h_f, w_f = frame.shape[:2]
                        ar = ((x2 - x1) * (y2 - y1)) / (w_f * h_f)
                        sev = compute_severity_score(conf_val, ar, cls)
                        prev_boxes.append((x1, y1, x2, y2, cls, conf_val, sev))

                total_detections_session += len(prev_boxes)

            # Draw detections
            class_counts = {}
            for (x1, y1, x2, y2, cls, conf_val, sev) in prev_boxes:
                draw_detection(frame, (x1, y1, x2, y2), cls, conf_val, sev)
                class_counts[cls] = class_counts.get(cls, 0) + 1

            # FPS
            elapsed = time.perf_counter() - t_start
            fps_buffer.append(1.0 / elapsed if elapsed > 0 else 0)
            fps = sum(fps_buffer) / len(fps_buffer)

            if show_fps:
                draw_hud(frame, fps, len(prev_boxes), class_counts, conf_threshold)

        cv2.imshow("🚗 Road Damage Detection — Live", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('+') or key == ord('='):
            conf_threshold = min(conf_threshold + 0.05, 0.95)
            print(f"📊 Confidence threshold: {conf_threshold:.2f}")
        elif key == ord('-'):
            conf_threshold = max(conf_threshold - 0.05, 0.05)
            print(f"📊 Confidence threshold: {conf_threshold:.2f}")
        elif key == ord('s'):
            screenshot_path = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"📸 Screenshot saved: {screenshot_path}")
        elif key == ord('p'):
            paused = not paused
            print("⏸ Paused" if paused else "▶️  Resumed")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n📊 Session complete — {total_detections_session} total detections")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time road damage detection")
    parser.add_argument("--source", default=0, help="Video source (0=webcam, or path to video)")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to YOLO model")
    args = parser.parse_args()

    MODEL_PATH = args.model
    run_realtime(source=args.source, conf_threshold=args.conf)
