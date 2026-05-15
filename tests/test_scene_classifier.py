"""Tests for scene risk classification."""

import pytest

from src.detection.schemas import Detection
from src.risk.scene_classifier import SceneRiskResult, classify_scene


def create_mock_scored_detection(
    class_name: str,
    risk_score: float,
    in_danger_zone: bool = False,
) -> Detection:
    """Helper to create a scored Detection object."""
    return Detection(
        class_name=class_name,
        confidence=0.9,
        bbox_xyxy=(0, 0, 100, 100),
        bbox_width=100,
        bbox_height=100,
        bbox_area_ratio=0.05,
        center=(50, 50),
        bottom_center=(50, 100),
        in_danger_zone=in_danger_zone,
        risk_score=risk_score,
    )


def test_classify_scene_empty():
    """Empty detections return LOW risk."""
    result = classify_scene([])
    assert isinstance(result, SceneRiskResult)
    assert result.risk_level == "LOW"
    assert result.max_risk_score == 0.0
    assert "No critical road users" in result.reason


def test_classify_scene_low():
    """Scene with only low-risk objects is LOW."""
    detections = [
        create_mock_scored_detection("car", 15.0),
        create_mock_scored_detection("person", 30.0),
    ]
    result = classify_scene(detections)
    assert result.risk_level == "LOW"
    assert result.max_risk_score == 30.0
    assert result.class_counts == {"car": 1, "person": 1}


def test_classify_scene_medium():
    """Scene with at least one medium-risk object is MEDIUM."""
    detections = [
        create_mock_scored_detection("car", 20.0),
        create_mock_scored_detection("bus", 45.0, in_danger_zone=False),
    ]
    result = classify_scene(detections)
    assert result.risk_level == "MEDIUM"
    assert result.max_risk_score == 45.0
    assert "Large vehicle" in result.reason


def test_classify_scene_high():
    """Scene with at least one high-risk object is HIGH."""
    detections = [
        create_mock_scored_detection("car", 20.0),
        create_mock_scored_detection("person", 85.0, in_danger_zone=True),
    ]
    result = classify_scene(detections)
    assert result.risk_level == "HIGH"
    assert result.max_risk_score == 85.0
    assert "CRITICAL" in result.reason
    assert "Person" in result.reason


def test_scene_result_to_dict():
    """SceneRiskResult serializes cleanly to dict."""
    detections = [create_mock_scored_detection("car", 20.0)]
    result = classify_scene(detections)
    d = result.to_dict()
    assert isinstance(d, dict)
    assert d["risk_level"] == "LOW"
    assert d["max_risk_score"] == 20.0
