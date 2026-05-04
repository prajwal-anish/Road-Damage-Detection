"""
Backend Test Suite — Road Damage Detection API
Run with: pytest tests/ -v
"""

import io
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def make_fake_image(width=640, height=480, color=(100, 100, 100)) -> bytes:
    """Create a minimal valid JPEG image as bytes."""
    import cv2
    img = np.full((height, width, 3), color, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def make_mock_yolo_result(detections: list[dict]):
    """
    Build a mock YOLO result mimicking ultralytics output structure.
    Each detection dict: {cls, conf, x1, y1, x2, y2}
    """
    result = MagicMock()

    if not detections:
        result.boxes = []
        return result

    boxes = MagicMock()
    boxes.__iter__ = MagicMock(return_value=iter([
        _make_box(d) for d in detections
    ]))
    boxes.__len__ = MagicMock(return_value=len(detections))
    result.boxes = boxes
    return result


def _make_box(d: dict):
    import torch
    box = MagicMock()
    box.xyxy = [torch.tensor([d["x1"], d["y1"], d["x2"], d["y2"]])]
    box.conf = [torch.tensor(d["conf"])]
    box.cls = [torch.tensor(d["cls"])]
    return box


# ─────────────────────────────────────────────
# Test: Severity Module
# ─────────────────────────────────────────────

class TestSeverityModule:
    def test_imports(self):
        from utils.severity import compute_severity, SeverityLevel, SEVERITY_THRESHOLDS
        assert callable(compute_severity)
        assert "LOW" in SEVERITY_THRESHOLDS
        assert "CRITICAL" in SEVERITY_THRESHOLDS

    def test_low_severity(self):
        from utils.severity import compute_severity
        label, score = compute_severity(confidence=0.3, area_ratio=0.005, class_id=2)
        assert label == "LOW"
        assert 0.0 <= score < 0.25

    def test_critical_severity_pothole(self):
        from utils.severity import compute_severity
        # High confidence + large area pothole = CRITICAL
        label, score = compute_severity(confidence=0.95, area_ratio=0.12, class_id=0)
        assert label in ("HIGH", "CRITICAL")
        assert score >= 0.5

    def test_manhole_lower_weight(self):
        from utils.severity import compute_severity
        _, pothole_score = compute_severity(confidence=0.8, area_ratio=0.05, class_id=0)
        _, manhole_score = compute_severity(confidence=0.8, area_ratio=0.05, class_id=2)
        assert pothole_score > manhole_score, "Potholes should score higher than manholes"

    def test_severity_level_from_score(self):
        from utils.severity import SeverityLevel
        assert SeverityLevel.from_score(0.0).label == "LOW"
        assert SeverityLevel.from_score(0.25).label == "MEDIUM"
        assert SeverityLevel.from_score(0.50).label == "HIGH"
        assert SeverityLevel.from_score(0.75).label == "CRITICAL"
        assert SeverityLevel.from_score(1.0).label == "CRITICAL"

    def test_score_clamped(self):
        from utils.severity import compute_severity
        _, score = compute_severity(confidence=1.0, area_ratio=1.0, class_id=0)
        assert 0.0 <= score <= 1.0

    def test_aggregate_severity(self):
        from utils.severity import aggregate_severity
        label, score = aggregate_severity([0.1, 0.8, 0.3, 0.9])
        assert label in ("HIGH", "CRITICAL")

    def test_aggregate_empty(self):
        from utils.severity import aggregate_severity
        label, score = aggregate_severity([])
        assert label == "LOW"
        assert score == 0.0

    def test_repair_recommendation(self):
        from utils.severity import get_repair_recommendation
        rec = get_repair_recommendation("CRITICAL", "pothole")
        assert "action" in rec
        assert "timeline" in rec
        assert "priority" in rec
        assert rec["priority"] == 1

    @pytest.mark.parametrize("cls_id,expected_weight", [
        (0, 1.0),
        (1, 0.75),
        (2, 0.4),
    ])
    def test_class_weights(self, cls_id, expected_weight):
        from utils.severity import CLASS_SEVERITY_WEIGHTS
        assert CLASS_SEVERITY_WEIGHTS[cls_id] == expected_weight


# ─────────────────────────────────────────────
# Test: Logger Module
# ─────────────────────────────────────────────

class TestLoggerModule:
    def test_setup_logger(self, tmp_path):
        from utils.logger import setup_logger
        log_file = tmp_path / "test.log"
        logger = setup_logger("test_logger", log_file)
        assert logger is not None
        logger.info("Test message")
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

    def test_log_detection_event(self, tmp_path):
        from utils.logger import setup_logger, log_detection_event
        import os
        os.chdir(tmp_path)
        (tmp_path / "logs").mkdir()

        logger = setup_logger("det_test", tmp_path / "test.log")
        log_detection_event(
            logger=logger,
            request_id="abc123",
            filename="road.jpg",
            num_detections=3,
            class_counts={"pothole": 2, "crack": 1},
            inference_ms=42.5,
            severity="HIGH",
        )

        jsonl_path = tmp_path / "logs" / "detections.jsonl"
        assert jsonl_path.exists()
        event = json.loads(jsonl_path.read_text().strip())
        assert event["request_id"] == "abc123"
        assert event["num_detections"] == 3
        assert event["severity"] == "HIGH"

    def test_load_detection_logs(self, tmp_path):
        from utils.logger import setup_logger, log_detection_event, load_detection_logs
        import os
        os.chdir(tmp_path)
        (tmp_path / "logs").mkdir()

        logger = setup_logger("load_test", tmp_path / "test.log")
        for i in range(3):
            log_detection_event(logger, f"req{i}", f"img{i}.jpg", i, {}, float(i*10), "LOW")

        logs = load_detection_logs(limit=10)
        assert len(logs) == 3
        # Most recent first
        assert logs[0]["request_id"] == "req2"


# ─────────────────────────────────────────────
# Test: FastAPI Endpoints
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create test client with mocked YOLO model."""
    with patch("main.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = [make_mock_yolo_result([
            {"cls": 0, "conf": 0.87, "x1": 100, "y1": 150, "x2": 300, "y2": 280},
            {"cls": 1, "conf": 0.72, "x1": 350, "y1": 200, "x2": 500, "y2": 310},
        ])]
        mock_load.return_value = mock_model

        # Patch global _model so startup doesn't fail
        with patch("main._model", mock_model):
            from main import app
            yield TestClient(app)


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert "message" in data
        assert "predict" in data["endpoints"]

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        assert isinstance(data["avg_inference_ms"], float)

    def test_stats_endpoint(self, client):
        r = client.get("/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total_inferences" in data
        assert "avg_inference_ms" in data
        assert "class_detections" in data


class TestPredictEndpoint:
    def test_valid_jpeg_returns_detections(self, client):
        img_bytes = make_fake_image()
        r = client.post(
            "/predict?include_annotated_image=false",
            files={"file": ("road.jpg", img_bytes, "image/jpeg")},
        )
        assert r.status_code == 200
        data = r.json()
        assert "detections" in data
        assert "summary" in data
        assert "inference_time_ms" in data
        assert isinstance(data["detections"], list)

    def test_detection_schema(self, client):
        img_bytes = make_fake_image()
        r = client.post(
            "/predict",
            files={"file": ("road.jpg", img_bytes, "image/jpeg")},
        )
        assert r.status_code == 200
        data = r.json()
        if data["detections"]:
            det = data["detections"][0]
            required_keys = {"id", "class_id", "class_name", "confidence", "bbox",
                             "area_px", "area_ratio", "severity", "severity_score", "color"}
            assert required_keys.issubset(det.keys())
            bbox_keys = {"x1", "y1", "x2", "y2", "width", "height", "center_x", "center_y"}
            assert bbox_keys.issubset(det["bbox"].keys())

    def test_summary_schema(self, client):
        img_bytes = make_fake_image()
        r = client.post(
            "/predict?include_annotated_image=false",
            files={"file": ("road.jpg", img_bytes, "image/jpeg")},
        )
        assert r.status_code == 200
        summary = r.json()["summary"]
        assert "total_detections" in summary
        assert "class_counts" in summary
        assert "overall_severity" in summary
        assert "affected_area_ratio" in summary
        assert summary["overall_severity"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_confidence_values_in_range(self, client):
        img_bytes = make_fake_image()
        r = client.post(
            "/predict",
            files={"file": ("road.jpg", img_bytes, "image/jpeg")},
        )
        for det in r.json()["detections"]:
            assert 0.0 <= det["confidence"] <= 1.0
            assert 0.0 <= det["severity_score"] <= 1.0

    def test_custom_conf_threshold(self, client):
        img_bytes = make_fake_image()
        r = client.post(
            "/predict?conf=0.8",
            files={"file": ("road.jpg", img_bytes, "image/jpeg")},
        )
        assert r.status_code == 200
        assert r.json()["model_confidence_threshold"] == 0.8

    def test_annotated_image_included(self, client):
        img_bytes = make_fake_image()
        r = client.post(
            "/predict?include_annotated_image=true",
            files={"file": ("road.jpg", img_bytes, "image/jpeg")},
        )
        assert r.status_code == 200
        b64 = r.json().get("annotated_image_base64")
        assert b64 is not None
        import base64
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_invalid_file_type_rejected(self, client):
        r = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"fake pdf content", "application/pdf")},
        )
        assert r.status_code == 400
        assert "Invalid file type" in r.json()["detail"]

    def test_invalid_conf_range(self, client):
        img_bytes = make_fake_image()
        r = client.post(
            "/predict?conf=1.5",
            files={"file": ("road.jpg", img_bytes, "image/jpeg")},
        )
        assert r.status_code == 422  # Validation error

    def test_png_image_accepted(self, client):
        import cv2
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        r = client.post(
            "/predict",
            files={"file": ("road.png", buf.tobytes(), "image/png")},
        )
        assert r.status_code == 200

    def test_request_id_unique(self, client):
        img_bytes = make_fake_image()
        ids = set()
        for _ in range(3):
            r = client.post(
                "/predict?include_annotated_image=false",
                files={"file": ("road.jpg", img_bytes, "image/jpeg")},
            )
            ids.add(r.json()["request_id"])
        assert len(ids) == 3, "Request IDs must be unique"

    def test_image_dimensions_in_response(self, client):
        img_bytes = make_fake_image(width=800, height=600)
        r = client.post(
            "/predict?include_annotated_image=false",
            files={"file": ("road.jpg", img_bytes, "image/jpeg")},
        )
        data = r.json()
        assert data["image_width"] == 800
        assert data["image_height"] == 600


class TestBatchEndpoint:
    def test_batch_multiple_images(self, client):
        files = [
            ("files", ("img1.jpg", make_fake_image(), "image/jpeg")),
            ("files", ("img2.jpg", make_fake_image(color=(200, 100, 50)), "image/jpeg")),
        ]
        r = client.post("/predict/batch", files=files)
        assert r.status_code == 200
        data = r.json()
        assert data["batch_size"] == 2
        assert len(data["results"]) == 2

    def test_batch_too_many_images(self, client):
        files = [("files", (f"img{i}.jpg", make_fake_image(), "image/jpeg")) for i in range(11)]
        r = client.post("/predict/batch", files=files)
        assert r.status_code == 400
        assert "10" in r.json()["detail"]


# ─────────────────────────────────────────────
# Test: No-Detection Edge Case
# ─────────────────────────────────────────────

@pytest.fixture
def client_no_dets():
    with patch("main.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = [make_mock_yolo_result([])]
        mock_load.return_value = mock_model
        with patch("main._model", mock_model):
            from main import app
            from importlib import reload
            yield TestClient(app)


class TestNoDetections:
    def test_zero_detections_summary(self, client_no_dets):
        img_bytes = make_fake_image()
        r = client_no_dets.post(
            "/predict?include_annotated_image=false",
            files={"file": ("clean_road.jpg", img_bytes, "image/jpeg")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["summary"]["total_detections"] == 0
        assert data["summary"]["overall_severity"] == "LOW"
        assert data["summary"]["affected_area_ratio"] == 0.0
        assert data["detections"] == []
