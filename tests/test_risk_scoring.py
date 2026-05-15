"""Tests for risk scoring."""

import pytest

from src.detection.schemas import Detection
from src.risk.scoring import calculate_object_risk, get_risk_reason


def create_mock_detection(
    class_name: str,
    y2: int = 100,
    area_ratio: float = 0.05,
    in_danger_zone: bool = False,
    confidence: float = 0.8,
) -> Detection:
    """Helper to create a Detection object for testing."""
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


def test_calculate_risk_pedestrian_in_zone():
    """Vulnerable user in danger zone produces very high risk."""
    det = create_mock_detection(
        class_name="person",
        y2=600,
        area_ratio=0.15,
        in_danger_zone=True,
    )
    score = calculate_object_risk(det, frame_height=1000)
    
    # Base: area_factor(15) + y_factor(600/1000*15 = 9) = 24
    # Penalty: danger_zone(30) + vulnerable(25) = 55
    # Conf: (0.8 - 0.5) * 10 = 3
    # Total ~ 82 (HIGH)
    assert score >= 70.0


def test_calculate_risk_distant_car():
    """Distant non-vulnerable object outside zone produces low risk."""
    det = create_mock_detection(
        class_name="car",
        y2=200,
        area_ratio=0.01,
        in_danger_zone=False,
    )
    score = calculate_object_risk(det, frame_height=1000)
    
    # Base: area_factor(1) + y_factor(200/1000*15 = 3) = 4
    # Conf: 3
    # Total ~ 7 (LOW)
    assert score < 35.0


def test_calculate_risk_large_vehicle_near_zone():
    """Large vehicle blocking lane gets penalty."""
    det = create_mock_detection(
        class_name="bus",
        y2=800,  # y_ratio > 0.5
        area_ratio=0.25,
        in_danger_zone=False,
    )
    score = calculate_object_risk(det, frame_height=1000)
    
    # Base: area(20 max) + y(12) = 32
    # Penalty: large_vehicle(10) since y > 0.5
    # Conf: 3
    # Total ~ 45 (MEDIUM)
    assert 35.0 <= score < 70.0


def test_risk_reasons():
    """Reasons correctly reflect the object state."""
    det1 = create_mock_detection("person", in_danger_zone=True)
    reason1 = get_risk_reason(det1)
    assert "Vulnerable user in driving lane" in reason1
    
    det2 = create_mock_detection("bus", in_danger_zone=True)
    reason2 = get_risk_reason(det2)
    assert "Large vehicle blocking driving lane" in reason2
    
    det3 = create_mock_detection("car", in_danger_zone=False)
    reason3 = get_risk_reason(det3)
    assert "Object detected (outside lane)" in reason3
