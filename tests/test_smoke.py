"""Smoke tests — verify imports and basic detector behaviour."""

import numpy as np
import pytest


# ── Import tests ────────────────────────────────────────────────


def test_detection_schema_import():
    """Detection dataclass can be imported and instantiated."""
    from src.detection.schemas import Detection

    det = Detection(
        class_name="car",
        confidence=0.92,
        bbox_xyxy=(100, 200, 300, 400),
        bbox_width=200,
        bbox_height=200,
        bbox_area_ratio=0.05,
        center=(200, 300),
        bottom_center=(200, 400),
    )
    assert det.class_name == "car"
    assert det.confidence == 0.92
    assert det.bbox_width == 200
    assert det.bbox_height == 200
    assert det.in_danger_zone is False
    assert det.risk_score == 0.0


def test_detection_with_risk():
    """with_risk() returns a new Detection with updated risk fields."""
    from src.detection.schemas import Detection

    det = Detection(
        class_name="person",
        confidence=0.88,
        bbox_xyxy=(50, 100, 150, 300),
        bbox_width=100,
        bbox_height=200,
        bbox_area_ratio=0.03,
        center=(100, 200),
        bottom_center=(100, 300),
    )
    updated = det.with_risk(in_danger_zone=True, risk_score=75.0, risk_reason="test")
    assert updated.in_danger_zone is True
    assert updated.risk_score == 75.0
    assert updated.risk_reason == "test"
    # Original is unchanged (frozen).
    assert det.in_danger_zone is False


def test_detection_to_dict():
    """to_dict() returns a plain dictionary."""
    from src.detection.schemas import Detection

    det = Detection(
        class_name="truck",
        confidence=0.71,
        bbox_xyxy=(0, 0, 50, 50),
        bbox_width=50,
        bbox_height=50,
        bbox_area_ratio=0.01,
        center=(25, 25),
        bottom_center=(25, 50),
    )
    d = det.to_dict()
    assert isinstance(d, dict)
    assert d["class_name"] == "truck"
    assert "bbox_xyxy" in d


def test_config_target_classes():
    """Config exposes target classes list."""
    from src.config import TARGET_CLASSES

    assert "person" in TARGET_CLASSES
    assert "car" in TARGET_CLASSES
    assert "truck" in TARGET_CLASSES
    assert len(TARGET_CLASSES) >= 6


# ── Detector tests (require model weights) ─────────────────────


def _model_available() -> bool:
    """Check if YOLO model weights can be downloaded/loaded."""
    try:
        from ultralytics import YOLO
        YOLO("yolov8n.pt")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _model_available(), reason="YOLO weights not available")
class TestRoadDetector:
    """Tests that require the YOLO model to be loadable."""

    def test_detector_instantiation(self):
        """RoadDetector loads without error."""
        from src.detection.detector import RoadDetector

        detector = RoadDetector()
        assert detector.model is not None

    def test_detect_returns_list(self):
        """detect() returns a list even on a blank image."""
        from src.detection.detector import RoadDetector

        detector = RoadDetector()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(blank)
        assert isinstance(results, list)

    def test_detect_empty_image(self):
        """detect() handles zero-size array gracefully."""
        from src.detection.detector import RoadDetector

        detector = RoadDetector()
        empty = np.array([], dtype=np.uint8)
        results = detector.detect(empty)
        assert results == []

    def test_class_filtering(self):
        """Only target classes survive filtering."""
        from src.detection.detector import RoadDetector
        from src.config import TARGET_CLASSES

        detector = RoadDetector()
        # Use a realistic-sized dummy image — detections are unlikely
        # but any that occur must be in TARGET_CLASSES.
        dummy = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        results = detector.detect(dummy)
        for det in results:
            assert det.class_name in TARGET_CLASSES

    def test_detection_fields(self):
        """Each detection has all required fields populated."""
        from src.detection.detector import RoadDetector

        detector = RoadDetector()
        dummy = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        results = detector.detect(dummy)
        for det in results:
            assert isinstance(det.class_name, str)
            assert 0.0 <= det.confidence <= 1.0
            assert len(det.bbox_xyxy) == 4
            assert det.bbox_width >= 0
            assert det.bbox_height >= 0
            assert 0.0 <= det.bbox_area_ratio <= 1.0
            assert len(det.center) == 2
            assert len(det.bottom_center) == 2
