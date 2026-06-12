"""
Detection — Data Schemas

Dataclasses for representing detection results.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Detection:
    """A single detected object in a frame.

    All spatial values are in pixel coordinates unless noted otherwise.

    Attributes:
        class_name:     COCO class label (e.g. "person", "car").
        confidence:     Detector confidence score, 0.0–1.0.
        bbox_xyxy:      Bounding box as (x1, y1, x2, y2).
        bbox_width:     Width of the bounding box in pixels.
        bbox_height:    Height of the bounding box in pixels.
        bbox_area_ratio: Bounding-box area / frame area (0.0–1.0).
        center:         Center point (cx, cy) of the bounding box.
        bottom_center:  Bottom-center point (cx, y2) — used for zone checks.
        in_danger_zone: Whether *bottom_center* is inside the danger-zone polygon.
                        Set downstream by the risk module; defaults to False.
        risk_score:     Composite risk score (0–100).
                        Set downstream by the risk module; defaults to 0.0.
        risk_reason:    Human-readable explanation of the risk score.
                        Set downstream by the risk module; defaults to "".
    """

    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    bbox_width: int
    bbox_height: int
    bbox_area_ratio: float
    center: tuple[int, int]
    bottom_center: tuple[int, int]

    # Populated later by the risk-scoring module.
    in_danger_zone: bool = field(default=False)
    risk_score: float = field(default=0.0)
    risk_reason: str = field(default="")
    
    # Populated by the depth estimation module if enabled
    estimated_depth: float = field(default=0.0)

    # ── helpers ─────────────────────────────────────────────────

    def with_risk(
        self,
        *,
        in_danger_zone: bool | None = None,
        risk_score: float | None = None,
        risk_reason: str | None = None,
        estimated_depth: float | None = None,
    ) -> Detection:
        """Return a new Detection with updated risk fields.

        Because Detection is frozen, this creates a shallow copy with the
        specified risk fields replaced.
        """
        from dataclasses import asdict

        data = asdict(self)
        if in_danger_zone is not None:
            data["in_danger_zone"] = in_danger_zone
        if risk_score is not None:
            data["risk_score"] = risk_score
        if risk_reason is not None:
            data["risk_reason"] = risk_reason
        if estimated_depth is not None:
            data["estimated_depth"] = estimated_depth
        return Detection(**data)

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary (JSON-friendly)."""
        from dataclasses import asdict

        return asdict(self)
