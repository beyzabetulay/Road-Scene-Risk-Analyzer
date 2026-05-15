"""Tests for export functionalities."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from src.detection.schemas import Detection
from src.io.exporters import (
    export_image_to_bytes,
    export_report_to_json,
    export_table_to_csv,
    generate_export_filename,
)
from src.pipeline import AnalysisResult
from src.risk.scene_classifier import SceneRiskResult


@pytest.fixture
def sample_result() -> AnalysisResult:
    det = Detection(
        class_name="car",
        confidence=0.9,
        bbox_xyxy=(10, 10, 50, 50),
        bbox_width=40,
        bbox_height=40,
        bbox_area_ratio=0.01,
        center=(30, 30),
        bottom_center=(30, 50),
        in_danger_zone=True,
        risk_score=50.0,
        risk_reason="In zone",
    )
    scene = SceneRiskResult(
        risk_level="MEDIUM",
        max_risk_score=50.0,
        reason="Test",
        class_counts={"car": 1},
    )
    return AnalysisResult(
        image_width=100,
        image_height=100,
        channels=3,
        detections=[det],
        detection_count=1,
        scene_risk=scene,
        timestamp="2023-01-01T00:00:00Z",
        settings={},
    )


def test_generate_export_filename():
    with patch("src.io.exporters.datetime") as mock_dt:
        # Mock datetime to return a fixed time
        mock_dt.now.return_value.strftime.return_value = "20230101_120000"
        fname = generate_export_filename("report", "HIGH", ".json")
        assert fname == "report_HIGH_20230101_120000.json"


def test_export_report_to_json(sample_result):
    json_str = export_report_to_json(sample_result)
    parsed = json.loads(json_str)
    assert parsed["image_width"] == 100
    assert parsed["scene_risk"]["risk_level"] == "MEDIUM"
    assert len(parsed["detections"]) == 1


def test_export_table_to_csv(sample_result):
    csv_str = export_table_to_csv(sample_result)
    lines = csv_str.strip().split("\r\n")
    assert len(lines) == 2  # Header + 1 detection
    assert "Class,Confidence,Risk Score,In Danger Zone,Reason" in lines[0]
    assert "car,0.9000,50.0,True,In zone" in lines[1]


def test_export_image_to_bytes():
    dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
    byte_data = export_image_to_bytes(dummy_image, ".png")
    assert isinstance(byte_data, bytes)
    assert len(byte_data) > 0
    
    # Invalid extension should raise
    with pytest.raises(Exception):
        export_image_to_bytes(dummy_image, ".invalid_ext")
