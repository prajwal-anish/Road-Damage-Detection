#!/usr/bin/env python3
"""
API Smoke Test Script
Quickly verifies the backend is running and returning valid responses.

Usage:
    python api_test.py                         # Uses localhost:8000
    python api_test.py --url https://my-api.railway.app
    python api_test.py --image /path/to/road.jpg
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

try:
    import requests
    import numpy as np
    import cv2
except ImportError:
    print("❌ Missing deps. Run: pip install requests opencv-python-headless numpy")
    sys.exit(1)


# ─────────────────────────────────────────────
# Test utilities
# ─────────────────────────────────────────────

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = {"passed": 0, "failed": 0, "warnings": 0}


def check(name: str, condition: bool, detail: str = "", warn_only: bool = False):
    if condition:
        print(f"  {PASS} {name}")
        results["passed"] += 1
    elif warn_only:
        print(f"  {WARN} {name}: {detail}")
        results["warnings"] += 1
    else:
        print(f"  {FAIL} {name}: {detail}")
        results["failed"] += 1


def make_test_image(w: int = 640, h: int = 480) -> bytes:
    """Create a simple test image resembling a road surface."""
    img = np.full((h, w, 3), (60, 55, 48), dtype=np.uint8)
    # Road markings
    cv2.rectangle(img, (w//3, 0), (2*w//3, h), (70, 65, 55), -1)
    cv2.line(img, (w//2, 0), (w//2, h), (90, 85, 70), 2)
    # Simulated damage circles
    cv2.ellipse(img, (150, 200), (50, 35), 0, 0, 360, (40, 35, 28), -1)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


# ─────────────────────────────────────────────
# Test suites
# ─────────────────────────────────────────────

def test_health(base_url: str) -> bool:
    print("\n📋 Health Check")
    print("-" * 40)
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        check("HTTP 200", r.status_code == 200, f"Got {r.status_code}")
        data = r.json()
        check("Has status field", "status" in data)
        check("Model loaded", data.get("model_loaded", False),
              "Model not loaded — place best.pt in model/ directory")
        check("Has uptime", "uptime_seconds" in data)
        check("Has avg_inference_ms", "avg_inference_ms" in data)
        if data.get("model_loaded"):
            print(f"     Model: {data.get('model_path', 'N/A')}")
        return data.get("model_loaded", False)
    except requests.exceptions.ConnectionError:
        print(f"  {FAIL} Cannot connect to {base_url}")
        print(f"       Is the backend running? Try: uvicorn main:app --reload")
        return False
    except Exception as e:
        print(f"  {FAIL} Health check error: {e}")
        return False


def test_root(base_url: str):
    print("\n📋 Root Endpoint")
    print("-" * 40)
    r = requests.get(f"{base_url}/", timeout=10)
    check("HTTP 200", r.status_code == 200)
    data = r.json()
    check("Has endpoints", "endpoints" in data)
    check("Has predict endpoint", "predict" in data.get("endpoints", {}))


def test_stats(base_url: str):
    print("\n📋 Stats Endpoint")
    print("-" * 40)
    r = requests.get(f"{base_url}/stats", timeout=10)
    check("HTTP 200", r.status_code == 200)
    data = r.json()
    check("Has total_inferences", "total_inferences" in data)
    check("Has class_detections", "class_detections" in data)
    classes = data.get("class_detections", {})
    for cls in ["pothole", "crack", "manhole"]:
        check(f"Has class '{cls}'", cls in classes, warn_only=True)


def test_predict(base_url: str, image_path: str = None):
    print("\n📋 Prediction Endpoint")
    print("-" * 40)

    if image_path and Path(image_path).exists():
        print(f"  Using image: {image_path}")
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        fname = Path(image_path).name
    else:
        print("  Using synthetic test image (640×480)")
        img_bytes = make_test_image()
        fname = "test_road.jpg"

    t0 = time.time()
    r = requests.post(
        f"{base_url}/predict",
        files={"file": (fname, img_bytes, "image/jpeg")},
        params={"conf": 0.25, "iou": 0.45, "include_annotated_image": True},
        timeout=60,
    )
    elapsed_ms = (time.time() - t0) * 1000

    check("HTTP 200", r.status_code == 200, f"Got {r.status_code}: {r.text[:200]}")
    if r.status_code != 200:
        return

    data = r.json()
    print(f"\n  📊 Response Summary:")
    print(f"     Request ID       : {data.get('request_id', 'N/A')}")
    print(f"     Image size       : {data.get('image_width')}×{data.get('image_height')}px")
    print(f"     Detections       : {data['summary']['total_detections']}")
    print(f"     Overall severity : {data['summary']['overall_severity']}")
    print(f"     Inference time   : {data.get('inference_time_ms', 0):.1f}ms")
    print(f"     Round-trip time  : {elapsed_ms:.1f}ms")
    print(f"     Class counts     : {data['summary']['class_counts']}")

    # Schema checks
    check("Has request_id", "request_id" in data)
    check("Has timestamp", "timestamp" in data)
    check("Has detections list", isinstance(data.get("detections"), list))
    check("Has summary", "summary" in data)
    check("Has inference_time_ms", isinstance(data.get("inference_time_ms"), (int, float)))
    check("Inference under 5s", data.get("inference_time_ms", 99999) < 5000,
          f"{data.get('inference_time_ms')}ms — model may be on CPU")

    # Annotated image
    b64 = data.get("annotated_image_base64")
    if b64:
        check("Annotated image is valid base64", True)
        decoded = base64.b64decode(b64)
        check("Annotated image non-empty", len(decoded) > 0, f"{len(decoded)} bytes")
    else:
        check("Annotated image present", False, warn_only=True)

    # Per-detection checks
    for i, det in enumerate(data["detections"]):
        prefix = f"det[{i}]"
        check(f"{prefix} has class_name", "class_name" in det)
        check(f"{prefix} class is valid",
              det.get("class_name") in ("pothole", "crack", "manhole"),
              det.get("class_name"))
        check(f"{prefix} conf in [0,1]", 0 <= det.get("confidence", -1) <= 1)
        check(f"{prefix} severity valid",
              det.get("severity") in ("LOW", "MEDIUM", "HIGH", "CRITICAL"))
        bbox = det.get("bbox", {})
        check(f"{prefix} bbox x2 > x1", bbox.get("x2", 0) > bbox.get("x1", 0))
        check(f"{prefix} bbox y2 > y1", bbox.get("y2", 0) > bbox.get("y1", 0))


def test_error_handling(base_url: str):
    print("\n📋 Error Handling")
    print("-" * 40)

    # Invalid file type
    r = requests.post(
        f"{base_url}/predict",
        files={"file": ("bad.pdf", b"fake", "application/pdf")},
        timeout=10,
    )
    check("Rejects PDF (400)", r.status_code == 400, f"Got {r.status_code}")

    # Out of range conf
    r = requests.post(
        f"{base_url}/predict?conf=1.5",
        files={"file": ("test.jpg", make_test_image(), "image/jpeg")},
        timeout=10,
    )
    check("Rejects conf=1.5 (422)", r.status_code == 422, f"Got {r.status_code}")

    # Corrupt image
    r = requests.post(
        f"{base_url}/predict",
        files={"file": ("corrupt.jpg", b"not an image at all", "image/jpeg")},
        timeout=10,
    )
    check("Rejects corrupt image (400/422)", r.status_code in (400, 422), f"Got {r.status_code}")


def test_performance(base_url: str, iterations: int = 5):
    print(f"\n📋 Performance Test ({iterations} iterations)")
    print("-" * 40)
    img_bytes = make_test_image()
    times = []

    for i in range(iterations):
        t0 = time.time()
        r = requests.post(
            f"{base_url}/predict",
            files={"file": ("perf.jpg", img_bytes, "image/jpeg")},
            params={"include_annotated_image": False},
            timeout=60,
        )
        elapsed = (time.time() - t0) * 1000
        if r.status_code == 200:
            inf_ms = r.json().get("inference_time_ms", 0)
            times.append(elapsed)
            print(f"  Run {i+1}: round-trip={elapsed:.0f}ms, inference={inf_ms:.0f}ms")
        else:
            print(f"  Run {i+1}: FAILED ({r.status_code})")

    if times:
        print(f"\n  📊 Performance Summary:")
        print(f"     Min    : {min(times):.0f}ms")
        print(f"     Max    : {max(times):.0f}ms")
        avg = sum(times) / len(times)
        print(f"     Avg    : {avg:.0f}ms")
        check("Avg round-trip < 10s", avg < 10000, f"{avg:.0f}ms")
        check("Avg round-trip < 2s", avg < 2000, f"{avg:.0f}ms", warn_only=True)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Road Damage API Smoke Test")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--image", default=None, help="Path to test image (optional)")
    parser.add_argument("--perf", action="store_true", help="Run performance benchmark")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print("=" * 60)
    print(f"🔍 Road Damage API Test Suite")
    print(f"   Target: {base_url}")
    print("=" * 60)

    model_ready = test_health(base_url)
    if not model_ready:
        print("\n⚠️  Skipping prediction tests — model not loaded.")
        print("   Place best.pt in model/ and restart the backend.")
    else:
        test_root(base_url)
        test_stats(base_url)
        test_predict(base_url, args.image)
        test_error_handling(base_url)
        if args.perf:
            test_performance(base_url)

    # Final summary
    print("\n" + "=" * 60)
    total = results["passed"] + results["failed"]
    print(f"📊 Results: {results['passed']}/{total} passed  |  "
          f"{results['warnings']} warnings  |  {results['failed']} failed")
    if results["failed"] == 0:
        print("🎉 All tests passed!")
    else:
        print(f"❌ {results['failed']} test(s) failed. Check output above.")
    print("=" * 60)

    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
