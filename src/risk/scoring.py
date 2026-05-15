"""
Risk — Scoring Engine

Computes per-object heuristic risk scores based on size,
position, object class, and danger zone presence.
"""

from __future__ import annotations

from src.detection.schemas import Detection

VULNERABLE_CLASSES = {"person", "bicycle", "motorcycle"}
LARGE_VEHICLE_CLASSES = {"bus", "truck"}


def calculate_object_risk(detection: Detection, frame_height: int) -> float:
    """Calculate a heuristic risk score (0-100) for a single detection.

    The score is composed of several components:
    - proximity: Based on bounding box area and how low it is in the frame.
    - danger_zone: Fixed penalty if the object is in the driving path.
    - vulnerable_user: Penalty for unprotected road users.
    - large_vehicle: Penalty for heavy vehicles near the bottom.
    - confidence: Small multiplier/boost for high-confidence detections.

    Args:
        detection:    The object detection result.
        frame_height: Height of the image frame in pixels.

    Returns:
        A float representing the risk score clamped between 0.0 and 100.0.
    """
    score = 0.0

    # 1. Proximity Component (0-35)
    # Area ratio (larger objects are usually closer).
    # Typical ratio is very small (e.g. 0.01 to 0.3).
    area_factor = min(detection.bbox_area_ratio * 100, 20.0)  # max 20 points
    
    # Y position (objects lower in the frame are closer).
    # y2 is the bottom of the bounding box.
    _, _, _, y2 = detection.bbox_xyxy
    y_ratio = y2 / frame_height if frame_height > 0 else 0.0
    y_factor = min(y_ratio * 15.0, 15.0)  # max 15 points
    
    score += area_factor + y_factor

    # 2. Danger Zone Component (+30)
    if detection.in_danger_zone:
        score += 30.0

    # 3. Vulnerable User Component (+25)
    if detection.class_name in VULNERABLE_CLASSES:
        score += 25.0

    # 4. Large Vehicle Component (+10)
    # If it's a truck/bus and it's fairly close (y_ratio > 0.5)
    if detection.class_name in LARGE_VEHICLE_CLASSES and y_ratio > 0.5:
        score += 10.0

    # 5. Confidence Component (+0-5)
    # Higher confidence slightly increases the risk certainty.
    confidence_factor = (detection.confidence - 0.5) * 10.0
    confidence_factor = max(0.0, min(confidence_factor, 5.0))
    score += confidence_factor

    # Clamp to 0-100
    return round(max(0.0, min(score, 100.0)), 2)


def get_risk_reason(detection: Detection) -> str:
    """Generate a human-readable reason string based on detection properties."""
    reasons = []

    if detection.in_danger_zone:
        if detection.class_name in VULNERABLE_CLASSES:
            reasons.append("Vulnerable user in driving lane")
        elif detection.class_name in LARGE_VEHICLE_CLASSES:
            reasons.append("Large vehicle blocking driving lane")
        else:
            reasons.append("Object in driving lane")
    else:
        if detection.class_name in VULNERABLE_CLASSES:
            reasons.append("Vulnerable user detected (outside lane)")
        elif detection.class_name in LARGE_VEHICLE_CLASSES:
            reasons.append("Large vehicle detected")
        else:
            reasons.append("Object detected (outside lane)")

    return " | ".join(reasons)
