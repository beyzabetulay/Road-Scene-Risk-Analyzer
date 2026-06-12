"""Tests for export functionalities (JSON, CSV, image bytes).

Covers:
- Filename generation with timestamp
- JSON export shape and content
- CSV export shape (header + rows) for image and video results
- Image encoding to bytes
- Edge cases (empty detections, invalid extensions)

Refs: #13
"""

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any
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
from src.pipeline import AnalysisResult, VideoAnalysisResult, VideoFrameResult
from src.risk.scene_classifier import SceneRiskResult


# ── Fixtures ───────────────────────────────────────────────────


def _make_detection(**overrides) -> Detection:
    defaults = dict(
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
    defaults.update(overrides)
    return Detection(**defaults)


@pytest.fixture
def sample_image_result() -> AnalysisResult:
    """An AnalysisResult with a single detection."""
    det = _make_detection()
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


@pytest.fixture
def sample_video_result() -> VideoAnalysisResult:
    """A VideoAnalysisResult with 2 frames of detections."""
    from src.io.media_loader import VideoInfo

    det1 = _make_detection(class_name="car", risk_score=30.0)
    det2 = _make_detection(class_name="person", risk_score=80.0, risk_reason="VRU in zone")

    frame_results = [
        VideoFrameResult(
            frame_index=0,
            detections=[det1],
            detection_count=1,
            max_risk_score=30.0,
            scene_risk_level="LOW",
        ),
        VideoFrameResult(
            frame_index=10,
            detections=[det2],
            detection_count=1,
            max_risk_score=80.0,
            scene_risk_level="HIGH",
        ),
    ]

    info = VideoInfo(
        path="/tmp/test_video.mp4",
        width=1280,
        height=720,
        fps=30.0,
        total_frames=300,
        duration_s=10.0,
    )

    return VideoAnalysisResult(
        video_info=info,
        frame_stride=10,
        frame_results=frame_results,
        total_frames_read=300,
        frames_processed=2,
        max_scene_risk=80.0,
        avg_object_count=1.0,
        high_risk_frames=1,
        riskiest_frame_index=10,
        timestamp="2023-01-01T00:00:00Z",
        settings={"model_name": "yolov8n.pt"},
    )


# ── Filename Generation Tests ─────────────────────────────────


class TestFilenameGeneration:
    """Tests for generate_export_filename()."""

    def test_filename_format(self):
        with patch("src.io.exporters.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20230101_120000"
            fname = generate_export_filename("report", "HIGH", ".json")
            assert fname == "report_HIGH_20230101_120000.json"

    def test_filename_with_csv_extension(self):
        with patch("src.io.exporters.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20230601_093000"
            fname = generate_export_filename("detections", "LOW", ".csv")
            assert fname == "detections_LOW_20230601_093000.csv"

    def test_filename_contains_risk_level(self):
        with patch("src.io.exporters.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20230101_000000"
            for level in ("LOW", "MEDIUM", "HIGH"):
                fname = generate_export_filename("x", level, ".json")
                assert level in fname


# ── JSON Export Tests ─────────────────────────────────────────


class TestJSONExport:
    """Tests for export_report_to_json()."""

    def test_json_is_valid(self, sample_image_result):
        json_str = export_report_to_json(sample_image_result)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_json_contains_image_dimensions(self, sample_image_result):
        parsed = json.loads(export_report_to_json(sample_image_result))
        assert parsed["image_width"] == 100
        assert parsed["image_height"] == 100

    def test_json_contains_scene_risk(self, sample_image_result):
        parsed = json.loads(export_report_to_json(sample_image_result))
        assert parsed["scene_risk"]["risk_level"] == "MEDIUM"
        assert parsed["scene_risk"]["max_risk_score"] == 50.0

    def test_json_contains_detections(self, sample_image_result):
        parsed = json.loads(export_report_to_json(sample_image_result))
        assert len(parsed["detections"]) == 1
        assert parsed["detections"][0]["class_name"] == "car"

    def test_json_empty_detections(self):
        """JSON export with zero detections."""
        scene = SceneRiskResult(
            risk_level="LOW", max_risk_score=0.0, reason="No objects"
        )
        result = AnalysisResult(
            image_width=640, image_height=480, channels=3,
            detections=[], detection_count=0, scene_risk=scene,
            timestamp="t", settings={},
        )
        parsed = json.loads(export_report_to_json(result))
        assert parsed["detections"] == []
        assert parsed["detection_count"] == 0


# ── CSV Export Tests ──────────────────────────────────────────


class TestCSVExport:
    """Tests for export_table_to_csv()."""

    def test_csv_image_header_and_rows(self, sample_image_result):
        csv_str = export_table_to_csv(sample_image_result)
        lines = csv_str.strip().split("\r\n")
        assert len(lines) == 2  # header + 1 detection
        assert "Class,Confidence,Risk Score,In Danger Zone,Reason" in lines[0]

    def test_csv_image_detection_values(self, sample_image_result):
        csv_str = export_table_to_csv(sample_image_result)
        lines = csv_str.strip().split("\r\n")
        assert "car,0.9000,50.0,True,In zone" in lines[1]

    def test_csv_video_has_frame_index_column(self, sample_video_result):
        csv_str = export_table_to_csv(sample_video_result)
        reader = csv.reader(io.StringIO(csv_str))
        header = next(reader)
        assert header[0] == "Frame Index"
        rows = list(reader)
        assert len(rows) == 2  # 2 frames, 1 detection each

    def test_csv_video_frame_indices(self, sample_video_result):
        csv_str = export_table_to_csv(sample_video_result)
        reader = csv.reader(io.StringIO(csv_str))
        next(reader)  # skip header
        rows = list(reader)
        assert rows[0][0] == "0"
        assert rows[1][0] == "10"

    def test_csv_empty_detections(self):
        """CSV with zero detections has only a header row."""
        scene = SceneRiskResult(
            risk_level="LOW", max_risk_score=0.0, reason="Empty"
        )
        result = AnalysisResult(
            image_width=640, image_height=480, channels=3,
            detections=[], detection_count=0, scene_risk=scene,
            timestamp="t", settings={},
        )
        csv_str = export_table_to_csv(result)
        lines = csv_str.strip().split("\r\n")
        assert len(lines) == 1  # header only


# ── Image Encoding Tests ─────────────────────────────────────


class TestImageExport:
    """Tests for export_image_to_bytes()."""

    def test_png_encoding(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        data = export_image_to_bytes(img, ".png")
        assert isinstance(data, bytes)
        assert len(data) > 0
        # PNG magic bytes
        assert data[:4] == b"\x89PNG"

    def test_jpg_encoding(self):
        img = np.ones((10, 10, 3), dtype=np.uint8) * 128
        data = export_image_to_bytes(img, ".jpg")
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_invalid_extension_raises(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(Exception):
            export_image_to_bytes(img, ".invalid_ext")
