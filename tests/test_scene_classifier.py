"""Tests for scene-level risk classification.

Covers:
- Empty detections → LOW
- LOW / MEDIUM / HIGH threshold behavior
- Class count aggregation
- SceneRiskResult serialization
- Reason string generation for all risk levels
- Mixed-class scenes

Refs: #13
"""

import pytest

from src.config import RISK_THRESHOLD_HIGH, RISK_THRESHOLD_LOW
from src.detection.schemas import Detection
from src.risk.scene_classifier import SceneRiskResult, classify_scene


# ── Test Helpers ────────────────────────────────────────────────


def _make_scored_detection(
    class_name: str = "car",
    risk_score: float = 10.0,
    in_danger_zone: bool = False,
    confidence: float = 0.9,
) -> Detection:
    """Create a Detection with a pre-set risk score."""
    return Detection(
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=(0, 0, 100, 100),
        bbox_width=100,
        bbox_height=100,
        bbox_area_ratio=0.05,
        center=(50, 50),
        bottom_center=(50, 100),
        in_danger_zone=in_danger_zone,
        risk_score=risk_score,
    )


# ── Threshold Behavior Tests ──────────────────────────────────


class TestClassifyScene:
    """Tests for classify_scene() threshold behavior."""

    def test_empty_detections_returns_low(self):
        """No detections → LOW with score 0.0."""
        result = classify_scene([])
        assert isinstance(result, SceneRiskResult)
        assert result.risk_level == "LOW"
        assert result.max_risk_score == 0.0
        assert "No critical road users" in result.reason

    def test_all_low_scores_returns_low(self):
        """All objects below LOW threshold → scene is LOW."""
        detections = [
            _make_scored_detection("car", risk_score=15.0),
            _make_scored_detection("person", risk_score=30.0),
        ]
        result = classify_scene(detections)
        assert result.risk_level == "LOW"
        assert result.max_risk_score == 30.0

    def test_score_at_low_threshold_returns_medium(self):
        """Score exactly at RISK_THRESHOLD_LOW → MEDIUM."""
        detections = [
            _make_scored_detection("car", risk_score=float(RISK_THRESHOLD_LOW)),
        ]
        result = classify_scene(detections)
        assert result.risk_level == "MEDIUM"

    def test_score_just_below_low_threshold_returns_low(self):
        """Score 1 point below LOW threshold → LOW."""
        detections = [
            _make_scored_detection("car", risk_score=float(RISK_THRESHOLD_LOW - 1)),
        ]
        result = classify_scene(detections)
        assert result.risk_level == "LOW"

    def test_score_at_high_threshold_returns_high(self):
        """Score exactly at RISK_THRESHOLD_HIGH → HIGH."""
        detections = [
            _make_scored_detection("person", risk_score=float(RISK_THRESHOLD_HIGH), in_danger_zone=True),
        ]
        result = classify_scene(detections)
        assert result.risk_level == "HIGH"

    def test_score_just_below_high_threshold_returns_medium(self):
        """Score 1 point below HIGH threshold → MEDIUM."""
        detections = [
            _make_scored_detection("car", risk_score=float(RISK_THRESHOLD_HIGH - 1)),
        ]
        result = classify_scene(detections)
        assert result.risk_level == "MEDIUM"

    def test_medium_range_scene(self):
        """A score in the MEDIUM range (35-69) → MEDIUM."""
        detections = [
            _make_scored_detection("car", risk_score=20.0),
            _make_scored_detection("bus", risk_score=45.0, in_danger_zone=False),
        ]
        result = classify_scene(detections)
        assert result.risk_level == "MEDIUM"
        assert result.max_risk_score == 45.0
        assert "Large vehicle" in result.reason

    def test_high_risk_scene(self):
        """A pedestrian with risk ≥70 in danger zone → HIGH."""
        detections = [
            _make_scored_detection("car", risk_score=20.0),
            _make_scored_detection("person", risk_score=85.0, in_danger_zone=True),
        ]
        result = classify_scene(detections)
        assert result.risk_level == "HIGH"
        assert result.max_risk_score == 85.0
        assert "CRITICAL" in result.reason
        assert "Person" in result.reason


# ── Class Counts Tests ─────────────────────────────────────────


class TestClassCounts:
    """Tests for detection class counting."""

    def test_class_counts_single(self):
        detections = [_make_scored_detection("car", 20.0)]
        result = classify_scene(detections)
        assert result.class_counts == {"car": 1}

    def test_class_counts_multiple(self):
        detections = [
            _make_scored_detection("car", 15.0),
            _make_scored_detection("car", 20.0),
            _make_scored_detection("person", 30.0),
        ]
        result = classify_scene(detections)
        assert result.class_counts == {"car": 2, "person": 1}

    def test_class_counts_empty(self):
        result = classify_scene([])
        assert result.class_counts == {}


# ── Reason Generation Tests ───────────────────────────────────


class TestSceneReasons:
    """Tests for scene-level reason strings."""

    def test_low_scene_few_objects(self):
        """LOW scene with ≤3 objects uses default reason."""
        detections = [_make_scored_detection("car", 10.0)]
        result = classify_scene(detections)
        assert "No critical road users" in result.reason

    def test_low_scene_many_objects(self):
        """LOW scene with >3 objects mentions multiple objects."""
        detections = [
            _make_scored_detection("car", 5.0),
            _make_scored_detection("car", 8.0),
            _make_scored_detection("car", 10.0),
            _make_scored_detection("car", 12.0),
        ]
        result = classify_scene(detections)
        assert "Multiple" in result.reason or "distant" in result.reason

    def test_high_scene_vru_in_zone_critical(self):
        """HIGH scene with VRU in zone → CRITICAL reason."""
        detections = [
            _make_scored_detection("person", 80.0, in_danger_zone=True),
        ]
        result = classify_scene(detections)
        assert "CRITICAL" in result.reason

    def test_medium_scene_large_vehicle_in_zone_warning(self):
        """MEDIUM scene with large vehicle in zone → WARNING reason."""
        detections = [
            _make_scored_detection("truck", 50.0, in_danger_zone=True),
        ]
        result = classify_scene(detections)
        assert "WARNING" in result.reason
        assert "truck" in result.reason

    def test_medium_scene_car_in_zone_warning(self):
        """MEDIUM scene with car in zone → WARNING reason."""
        detections = [
            _make_scored_detection("car", 50.0, in_danger_zone=True),
        ]
        result = classify_scene(detections)
        assert "WARNING" in result.reason
        assert "Car" in result.reason

    def test_high_scene_vru_not_in_zone(self):
        """VRU with high score but NOT in zone → nearby reason."""
        detections = [
            _make_scored_detection("person", 75.0, in_danger_zone=False),
        ]
        result = classify_scene(detections)
        assert "Vulnerable user" in result.reason
        assert "nearby" in result.reason


# ── Serialization Tests ────────────────────────────────────────


class TestSceneRiskResultSerialization:
    """Tests for SceneRiskResult.to_dict()."""

    def test_to_dict_has_required_keys(self):
        detections = [_make_scored_detection("car", 20.0)]
        result = classify_scene(detections)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "risk_level" in d
        assert "max_risk_score" in d
        assert "reason" in d
        assert "class_counts" in d

    def test_to_dict_values(self):
        detections = [_make_scored_detection("car", 20.0)]
        result = classify_scene(detections)
        d = result.to_dict()
        assert d["risk_level"] == "LOW"
        assert d["max_risk_score"] == 20.0
