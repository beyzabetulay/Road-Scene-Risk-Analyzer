"""Tests for image visualization and annotation."""

import numpy as np
import pytest

from src.detection.schemas import Detection
from src.pipeline import AnalysisResult
from src.risk.scene_classifier import SceneRiskResult
from src.visualization.annotator import annotate_image


def test_annotate_image_does_not_mutate():
    """Ensure annotate_image returns a new array and leaves original intact."""
    # Create a blank black image
    original_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Copy it to compare later
    image_copy = original_image.copy()

    det = Detection(
        class_name="person",
        confidence=0.9,
        bbox_xyxy=(100, 100, 200, 200),
        bbox_width=100,
        bbox_height=100,
        bbox_area_ratio=0.05,
        center=(150, 150),
        bottom_center=(150, 200),
        in_danger_zone=True,
        risk_score=85.0,
    )

    scene_risk = SceneRiskResult(
        risk_level="HIGH",
        max_risk_score=85.0,
        reason="CRITICAL: Person in driving lane",
        class_counts={"person": 1},
    )

    result = AnalysisResult(
        image_width=1280,
        image_height=720,
        channels=3,
        detections=[det],
        detection_count=1,
        scene_risk=scene_risk,
        timestamp="2023-01-01T00:00:00Z",
        settings={},
    )

    annotated = annotate_image(original_image, result, danger_zone=True)

    # Output should be the same shape and type
    assert annotated.shape == original_image.shape
    assert annotated.dtype == original_image.dtype

    # Output should NOT be the exact same object in memory
    assert id(annotated) != id(original_image)

    # Original should remain purely black (unchanged)
    np.testing.assert_array_equal(original_image, image_copy)

    # Annotated should have some non-zero pixels (drawings)
    assert np.any(annotated > 0)


def test_annotate_image_no_scene_risk():
    """Ensure it doesn't crash if scene_risk is somehow missing or None."""
    original_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    result = AnalysisResult(
        image_width=100,
        image_height=100,
        channels=3,
        detections=[],
        detection_count=0,
        scene_risk=None,  # type: ignore
        timestamp="",
        settings={},
    )
    
    # Should not raise
    annotated = annotate_image(original_image, result, danger_zone=False)
    assert annotated.shape == original_image.shape
