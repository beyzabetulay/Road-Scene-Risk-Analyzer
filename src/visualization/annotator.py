"""
Visualization — Annotator

Provides functions to overlay detections, risk scores,
and danger zones onto image arrays.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.config import RISK_THRESHOLD_HIGH, RISK_THRESHOLD_LOW
from src.pipeline import AnalysisResult
from src.risk.danger_zone import DangerZone


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
    result: AnalysisResult,
    *,
    draw_danger_zone: bool = True,
) -> np.ndarray:
    """Annotate an image with detections and scene risk information.

    Args:
        image: Original BGR image as a NumPy array.
        result: The AnalysisResult containing detections and scene risk.
        draw_danger_zone: If True, draws the danger zone polygon overlay.

    Returns:
        A new NumPy array containing the annotated image. Does not mutate
        the original image.
    """
    # Do not mutate the original image
    canvas = image.copy()
    h, w = canvas.shape[:2]

    # 1. Draw Danger Zone
    if draw_danger_zone:
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
        
        # Label text: "class (conf) - risk"
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
