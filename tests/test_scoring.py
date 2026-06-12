"""Tests for per-object risk scoring and reason strings.

Covers:
- Pedestrian in danger zone → HIGH risk (≥70)
- Distant car outside zone → LOW risk (<35)
- Large vehicle near lane → MEDIUM risk (35-69)
- Score clamping to 0-100
- Confidence component contribution
- Vulnerable user classes
- Reason string generation for all combinations

Refs: #13
"""

import pytest

from src.detection.schemas import Detection
from src.risk.scoring import (
    LARGE_VEHICLE_CLASSES,
    VULNERABLE_CLASSES,
    calculate_object_risk,
    get_risk_reason,
)


# ── Test Helpers ────────────────────────────────────────────────


def _make_detection(
    class_name: str = "car",
    y2: int = 100,
    area_ratio: float = 0.05,
    in_danger_zone: bool = False,
    confidence: float = 0.8,
) -> Detection:
    """Create a Detection object with sensible defaults for testing."""
    return Detection(
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=(0, 0, 100, y2),
        bbox_width=100,
        bbox_height=y2,
        bbox_area_ratio=area_ratio,
        center=(50, y2 // 2),
        bottom_center=(50, y2),
        in_danger_zone=in_danger_zone,
    )


# ── Score Calculation Tests ─────────────────────────────────────


class TestRiskScoreCalculation:
    """Tests for calculate_object_risk()."""

    def test_pedestrian_in_danger_zone_is_high_risk(self):
        """Acceptance: pedestrian inside danger-zone → HIGH (≥70)."""
        det = _make_detection(
            class_name="person",
            y2=600,
            area_ratio=0.15,
            in_danger_zone=True,
        )
        score = calculate_object_risk(det, frame_height=1000)
        assert score >= 70.0, f"Expected HIGH risk (≥70), got {score}"

    def test_distant_car_outside_zone_is_low_risk(self):
        """Acceptance: distant car outside zone → LOW (<35)."""
        det = _make_detection(
            class_name="car",
            y2=200,
            area_ratio=0.01,
            in_danger_zone=False,
        )
        score = calculate_object_risk(det, frame_height=1000)
        assert score < 35.0, f"Expected LOW risk (<35), got {score}"

    def test_distant_object_outside_zone_is_low_or_medium(self):
        """Acceptance: distant object outside danger-zone → LOW or MEDIUM."""
        det = _make_detection(
            class_name="car",
            y2=300,
            area_ratio=0.02,
            in_danger_zone=False,
        )
        score = calculate_object_risk(det, frame_height=1000)
        assert score < 70.0, f"Expected LOW or MEDIUM (<70), got {score}"

    def test_large_vehicle_near_lane_is_medium(self):
        """Large vehicle (bus) near the lane → MEDIUM (35-69)."""
        det = _make_detection(
            class_name="bus",
            y2=800,
            area_ratio=0.25,
            in_danger_zone=False,
        )
        score = calculate_object_risk(det, frame_height=1000)
        assert 35.0 <= score < 70.0, f"Expected MEDIUM risk (35-69), got {score}"

    def test_large_vehicle_far_no_penalty(self):
        """Bus far from the camera (y_ratio ≤ 0.5) gets no large-vehicle penalty."""
        det = _make_detection(
            class_name="bus",
            y2=400,          # y_ratio = 0.4
            area_ratio=0.03,
            in_danger_zone=False,
        )
        score = calculate_object_risk(det, frame_height=1000)
        # Without danger_zone and VRU bonus, and no large_vehicle penalty
        assert score < 35.0

    def test_bicycle_in_zone_is_high_risk(self):
        """Bicycle (VRU) in danger zone should produce HIGH risk."""
        det = _make_detection(
            class_name="bicycle",
            y2=700,
            area_ratio=0.10,
            in_danger_zone=True,
        )
        score = calculate_object_risk(det, frame_height=1000)
        assert score >= 70.0

    def test_motorcycle_in_zone_is_high_risk(self):
        """Motorcycle (VRU) in danger zone should produce HIGH risk."""
        det = _make_detection(
            class_name="motorcycle",
            y2=650,
            area_ratio=0.12,
            in_danger_zone=True,
        )
        score = calculate_object_risk(det, frame_height=1000)
        assert score >= 70.0

    def test_score_clamped_to_0_100(self):
        """Score should never exceed 100 even with all bonuses maxed."""
        det = _make_detection(
            class_name="person",
            y2=1000,
            area_ratio=0.50,
            in_danger_zone=True,
            confidence=1.0,
        )
        score = calculate_object_risk(det, frame_height=1000)
        assert 0.0 <= score <= 100.0

    def test_score_minimum_is_zero(self):
        """Minimal detection should produce a low but non-negative score."""
        det = _make_detection(
            class_name="car",
            y2=1,
            area_ratio=0.0001,
            in_danger_zone=False,
            confidence=0.25,
        )
        score = calculate_object_risk(det, frame_height=1000)
        assert score >= 0.0

    def test_confidence_component_contributes(self):
        """Higher confidence should give a slightly higher score (all else equal)."""
        det_low_conf = _make_detection(confidence=0.5)
        det_high_conf = _make_detection(confidence=1.0)
        score_low = calculate_object_risk(det_low_conf, frame_height=1000)
        score_high = calculate_object_risk(det_high_conf, frame_height=1000)
        assert score_high > score_low

    def test_danger_zone_adds_30_points(self):
        """Being in the danger zone adds exactly 30 points to the score."""
        det_out = _make_detection(in_danger_zone=False)
        det_in = _make_detection(in_danger_zone=True)
        score_out = calculate_object_risk(det_out, frame_height=1000)
        score_in = calculate_object_risk(det_in, frame_height=1000)
        assert score_in - score_out == pytest.approx(30.0)

    def test_vulnerable_user_adds_25_points(self):
        """VRU class adds 25 points vs non-VRU (same position, no zone)."""
        det_car = _make_detection(class_name="car", in_danger_zone=False)
        det_person = _make_detection(class_name="person", in_danger_zone=False)
        score_car = calculate_object_risk(det_car, frame_height=1000)
        score_person = calculate_object_risk(det_person, frame_height=1000)
        assert score_person - score_car == pytest.approx(25.0)

    def test_zero_frame_height_no_crash(self):
        """Edge case: frame_height=0 should not crash (ZeroDivisionError)."""
        det = _make_detection()
        score = calculate_object_risk(det, frame_height=0)
        assert score >= 0.0


# ── Risk Reason Tests ──────────────────────────────────────────


class TestRiskReasons:
    """Tests for get_risk_reason()."""

    def test_person_in_zone_reason(self):
        det = _make_detection("person", in_danger_zone=True)
        reason = get_risk_reason(det)
        assert "Vulnerable user in driving lane" in reason

    def test_bus_in_zone_reason(self):
        det = _make_detection("bus", in_danger_zone=True)
        reason = get_risk_reason(det)
        assert "Large vehicle" in reason
        assert "blocking driving lane" in reason

    def test_car_in_zone_reason(self):
        det = _make_detection("car", in_danger_zone=True)
        reason = get_risk_reason(det)
        assert "Object in driving lane" in reason

    def test_person_outside_zone_reason(self):
        det = _make_detection("person", in_danger_zone=False)
        reason = get_risk_reason(det)
        assert "Vulnerable user detected" in reason
        assert "outside lane" in reason

    def test_truck_outside_zone_reason(self):
        det = _make_detection("truck", in_danger_zone=False)
        reason = get_risk_reason(det)
        assert "Large vehicle detected" in reason

    def test_car_outside_zone_reason(self):
        det = _make_detection("car", in_danger_zone=False)
        reason = get_risk_reason(det)
        assert "Object detected (outside lane)" in reason


# ── Class Constants Tests ──────────────────────────────────────


class TestScoringConstants:
    """Validate that the class-group constants are correct."""

    def test_vulnerable_classes(self):
        assert "person" in VULNERABLE_CLASSES
        assert "bicycle" in VULNERABLE_CLASSES
        assert "motorcycle" in VULNERABLE_CLASSES
        assert "car" not in VULNERABLE_CLASSES

    def test_large_vehicle_classes(self):
        assert "bus" in LARGE_VEHICLE_CLASSES
        assert "truck" in LARGE_VEHICLE_CLASSES
        assert "car" not in LARGE_VEHICLE_CLASSES
