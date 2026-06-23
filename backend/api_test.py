
#!/usr/bin/env python3
"""
API Smoke Test Script
Quickly verifies the backend is running and returning valid responses.
"""

import argparse
import base64
import sys
import time
from pathlib import Path

import pytest

try:
    import requests
    import numpy as np
    import cv2
except ImportError:
    print("❌ Missing deps. Run: pip install requests opencv-python-headless numpy")
    sys.exit(1)


# ─────────────────────────────────────────────
# PYTEST FIXTURE
# ─────────────────────────────────────────────

@pytest.fixture
def base_url():
    return "http://localhost:8000"


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

    cv2.rectangle(img, (w // 3, 0), (2 * w // 3, h), (70, 65, 55), -1)
    cv2.line(img, (w // 2, 0), (w // 2, h), (90, 85, 70), 2)

    cv2.ellipse(
        img,
        (150, 200),
        (50, 35),
        0,
        0,
        360,
        (40, 35, 28),
        -1,
    )

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return buf.tobytes()


# ─────────────────────────────────────────────
# Test suites
# ─────────────────────────────────────────────

def test_health(base_url):
    print("\n📋 Health Check")
    print("-" * 40)

    try:
        r = requests.get(f"{base_url}/health", timeout=10)

        check("HTTP 200", r.status_code == 200)

        data = r.json()

        check("Has status field", "status" in data)
        check("Has uptime", "uptime_seconds" in data)
        check("Has avg_inference_ms", "avg_inference_ms" in data)

        return True

    except Exception as e:
        print(f"{FAIL} {e}")
        return False


def test_root(base_url):
    print("\n📋 Root Endpoint")
    print("-" * 40)

    r = requests.get(f"{base_url}/", timeout=10)

    check("HTTP 200", r.status_code == 200)

    data = r.json()

    check("Has endpoints", "endpoints" in data)


def test_stats(base_url):
    print("\n📋 Stats Endpoint")
    print("-" * 40)

    r = requests.get(f"{base_url}/stats", timeout=10)

    check("HTTP 200", r.status_code == 200)

    data = r.json()

    check("Has total_inferences", "total_inferences" in data)
    check("Has class_detections", "class_detections" in data)


def test_predict(base_url):
    print("\n📋 Prediction Endpoint")
    print("-" * 40)

    img_bytes = make_test_image()

    r = requests.post(
        f"{base_url}/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        timeout=60,
    )

    check("HTTP 200", r.status_code == 200)

    if r.status_code != 200:
        return

    data = r.json()

    check("Has request_id", "request_id" in data)
    check("Has detections", "detections" in data)
    check("Has summary", "summary" in data)


def test_error_handling(base_url):
    print("\n📋 Error Handling")
    print("-" * 40)

    r = requests.post(
        f"{base_url}/predict",
        files={"file": ("bad.pdf", b"fake", "application/pdf")},
        timeout=10,
    )

    check("Rejects PDF", r.status_code in [400, 422])


def test_performance(base_url):
    print("\n📋 Performance Test")
    print("-" * 40)

    img_bytes = make_test_image()

    t0 = time.time()

    r = requests.post(
        f"{base_url}/predict",
        files={"file": ("perf.jpg", img_bytes, "image/jpeg")},
        timeout=60,
    )

    elapsed = (time.time() - t0) * 1000

    check("HTTP 200", r.status_code == 200)

    print(f"Round-trip time: {elapsed:.0f}ms")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Road Damage API Smoke Test")

    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API base URL"
    )

    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print("=" * 60)
    print("🔍 Road Damage API Test Suite")
    print(f"Target: {base_url}")
    print("=" * 60)

    test_health(base_url)
    test_root(base_url)
    test_stats(base_url)
    test_predict(base_url)
    test_error_handling(base_url)
    test_performance(base_url)

    print("=" * 60)


if __name__ == "__main__":
    main()

