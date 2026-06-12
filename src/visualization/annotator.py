"""
Visualization — Annotator

Provides functions to overlay detections, risk scores,
and danger zones onto image arrays.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Any

from src.config import RISK_THRESHOLD_HIGH, RISK_THRESHOLD_LOW
from src.risk.danger_zone import DangerZone
from src.detection.schemas import Detection
from src.risk.scene_classifier import SceneRiskResult


# ── Colors (BGR format) ──
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 215, 255)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)


def _get_risk_color(score: float) -> tuple[int, int, int]:
    """Return BGR color based on risk score."""
    if score >= RISK_THRESHOLD_HIGH:
        return COLOR_RED
    if score >= RISK_THRESHOLD_LOW:
        return COLOR_YELLOW
    return COLOR_GREEN


def annotate_image(
    image: np.ndarray,
    result: Any,
    *,
    danger_zone: DangerZone | bool = True,
) -> np.ndarray:
    """Annotate an image with detections and scene risk information.

    Args:
        image: Original BGR image as a NumPy array.
        result: The AnalysisResult containing detections and scene risk.
        danger_zone: If True, draws the default danger zone polygon overlay.
                     If a DangerZone instance, draws that specific zone.
                     If False, skips drawing the zone.

    Returns:
        A new NumPy array containing the annotated image. Does not mutate
        the original image.
    """
    # Do not mutate the original image
    canvas = image.copy()
    h, w = canvas.shape[:2]

    # 1. Draw Danger Zone
    if danger_zone:
        if isinstance(danger_zone, DangerZone):
            zone = danger_zone
        else:
            zone = DangerZone(w, h)
        polygon = zone.get_polygon()
        # Draw translucent filled polygon
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [polygon], COLOR_BLUE)
        # Blend overlay with original
        cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
        # Draw bold outline
        cv2.polylines(canvas, [polygon], isClosed=True, color=COLOR_BLUE, thickness=2)

    # 2. Draw Detections
    for det in result.detections:
        color = _get_risk_color(det.risk_score)
        thickness = 3 if det.risk_score >= RISK_THRESHOLD_HIGH else 2
        
        x1, y1, x2, y2 = det.bbox_xyxy
        
        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        
        # Label text: "[ID] class (conf) - risk"
        if det.track_id is not None:
            label = f"[{det.track_id}] {det.class_name.upper()} ({det.confidence:.2f}) | Risk: {det.risk_score:.0f}"
        else:
            label = f"{det.class_name.upper()} ({det.confidence:.2f}) | Risk: {det.risk_score:.0f}"
        
        # Text background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Prevent drawing text box off the top of the image
        text_y = max(y1, text_h + 5)
        
        cv2.rectangle(
            canvas,
            (x1, text_y - text_h - 5),
            (x1 + text_w, text_y + 5),
            color,
            -1,
        )
        # Text
        cv2.putText(
            canvas,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_WHITE if color != COLOR_YELLOW else COLOR_BLACK,
            1,
        )

    # 3. Draw Scene Risk Panel
    if result.scene_risk:
        scene_color = _get_risk_color(result.scene_risk.max_risk_score)
        
        # Panel text
        panel_text1 = f"SCENE RISK: {result.scene_risk.risk_level}"
        panel_text2 = result.scene_risk.reason
        
        # Draw panel background at top-left
        cv2.rectangle(canvas, (10, 10), (w - 10, 70), (0, 0, 0), -1)
        cv2.rectangle(canvas, (10, 10), (w - 10, 70), scene_color, 2)
        
        # Draw texts
        cv2.putText(
            canvas,
            panel_text1,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            scene_color,
            2,
        )
        cv2.putText(
            canvas,
            panel_text2,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_WHITE,
            1,
        )

    return canvas

def annotate_video_frame(
    image: np.ndarray,
    detections: list[Detection],
    scene_risk: SceneRiskResult,
    *,
    danger_zone: DangerZone | bool = True,
) -> np.ndarray:
    """Annotate a single video frame with detections and scene risk.

    This is a lightweight version of `annotate_image` that accepts raw
    detections and scene risk, avoiding the need to construct a full
    AnalysisResult object for every frame of a video.

    Args:
        image: Original BGR image as a NumPy array.
        detections: List of Detection objects for this frame.
        scene_risk: The SceneRiskResult for this frame.
        danger_zone: If True, draws the default danger zone polygon overlay.
                     If a DangerZone instance, draws that specific zone.

    Returns:
        A new NumPy array containing the annotated image.
    """
    class DummyResult:
        def __init__(self, d, s):
            self.detections = d
            self.scene_risk = s

    return annotate_image(image, DummyResult(detections, scene_risk), danger_zone=danger_zone)
