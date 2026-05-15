"""
Risk — Scene Classifier

Classifies the overall road scene risk level based on the
aggregated risk scores of detected objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import RISK_THRESHOLD_HIGH, RISK_THRESHOLD_LOW
from src.detection.schemas import Detection
from src.risk.scoring import LARGE_VEHICLE_CLASSES, VULNERABLE_CLASSES


@dataclass
class SceneRiskResult:
    """Aggregate risk classification for an entire scene/frame.

    Attributes:
        risk_level:      "LOW", "MEDIUM", or "HIGH".
        max_risk_score:  The highest individual object risk score in the frame.
        reason:          Human-readable explanation of the scene classification.
        class_counts:    Dictionary counting occurrences of each object class.
    """

    risk_level: str
    max_risk_score: float
    reason: str
    class_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def classify_scene(detections: list[Detection]) -> SceneRiskResult:
    """Evaluate a list of detections to determine the overall scene risk.

    The scene risk score equals the highest object risk score.
    Classification levels are based on thresholds defined in config.py.

    Args:
        detections: List of Detection objects (already risk-scored).

    Returns:
        A :class:`SceneRiskResult` summarizing the scene's danger level.
    """
    if not detections:
        return SceneRiskResult(
            risk_level="LOW",
            max_risk_score=0.0,
            reason="No critical road users detected.",
        )

    # 1. Class Counts
    counts: dict[str, int] = {}
    for d in detections:
        counts[d.class_name] = counts.get(d.class_name, 0) + 1

    # 2. Find maximum risk
    riskiest_object = max(detections, key=lambda d: d.risk_score)
    max_score = riskiest_object.risk_score

    # 3. Determine level
    if max_score >= RISK_THRESHOLD_HIGH:
        level = "HIGH"
    elif max_score >= RISK_THRESHOLD_LOW:
        level = "MEDIUM"
    else:
        level = "LOW"

    # 4. Generate scene-level reason
    reason = _generate_scene_reason(level, riskiest_object, counts)

    return SceneRiskResult(
        risk_level=level,
        max_risk_score=max_score,
        reason=reason,
        class_counts=counts,
    )


def _generate_scene_reason(
    level: str,
    riskiest: Detection,
    counts: dict[str, int]
) -> str:
    """Generate a descriptive reason based on the worst offender and general counts."""
    if level == "LOW":
        if sum(counts.values()) > 3:
            return "Multiple distant or low-risk objects detected."
        return "No critical road users close to ego lane."

    # For HIGH or MEDIUM
    if riskiest.in_danger_zone:
        if riskiest.class_name in VULNERABLE_CLASSES:
            return f"CRITICAL: {riskiest.class_name.capitalize()} inside estimated driving lane!"
        if riskiest.class_name in LARGE_VEHICLE_CLASSES:
            return f"WARNING: Large vehicle ({riskiest.class_name}) blocking driving lane."
        return f"WARNING: {riskiest.class_name.capitalize()} in driving lane."
    else:
        # Not in danger zone but still high score (maybe very close)
        if riskiest.class_name in VULNERABLE_CLASSES:
            return f"Vulnerable user ({riskiest.class_name}) detected nearby."
        if riskiest.class_name in LARGE_VEHICLE_CLASSES:
            return f"Large vehicle ({riskiest.class_name}) close to ego lane."
        return f"Object ({riskiest.class_name}) detected in close proximity."
