"""Tests for danger zone mapping."""

import numpy as np
import pytest

from src.risk.danger_zone import DangerZone, DangerZoneParams


def test_danger_zone_polygon_shape():
    """DangerZone computes a polygon of the correct shape."""
    # Dummy params: square in the middle of a 100x100 frame
    params = DangerZoneParams(
        top_left=(0.25, 0.25),
        top_right=(0.75, 0.25),
        bottom_right=(0.75, 0.75),
        bottom_left=(0.25, 0.75),
    )
    zone = DangerZone(100, 100, params)
    poly = zone.get_polygon()
    
    assert isinstance(poly, np.ndarray)
    assert poly.shape == (4, 1, 2)
    assert poly.dtype == np.int32
    
    # Expected absolute points
    expected = [
        [[25, 75]],  # bottom_left
        [[25, 25]],  # top_left
        [[75, 25]],  # top_right
        [[75, 75]],  # bottom_right
    ]
    np.testing.assert_array_equal(poly, expected)


def test_danger_zone_contains_point():
    """DangerZone correctly identifies points inside and outside."""
    params = DangerZoneParams(
        top_left=(0.25, 0.25),
        top_right=(0.75, 0.25),
        bottom_right=(0.75, 0.75),
        bottom_left=(0.25, 0.75),
    )
    zone = DangerZone(100, 100, params)
    
    # Point clearly inside
    assert zone.contains_point((50, 50)) is True
    
    # Point clearly outside
    assert zone.contains_point((10, 10)) is False
    assert zone.contains_point((90, 90)) is False
    
    # Points exactly on the edge
    assert zone.contains_point((25, 50)) is True  # Left edge
    assert zone.contains_point((75, 50)) is True  # Right edge
    assert zone.contains_point((50, 25)) is True  # Top edge
    assert zone.contains_point((50, 75)) is True  # Bottom edge


def test_danger_zone_default_params():
    """DangerZone uses config defaults if no params are provided."""
    from src.config import DANGER_ZONE_BOTTOM_LEFT
    
    zone = DangerZone(1280, 720)
    poly = zone.get_polygon()
    
    # Check bottom_left point (first in the polygon array)
    expected_bl_x = int(DANGER_ZONE_BOTTOM_LEFT[0] * 1280)
    expected_bl_y = int(DANGER_ZONE_BOTTOM_LEFT[1] * 720)
    
    assert poly[0, 0, 0] == expected_bl_x
    assert poly[0, 0, 1] == expected_bl_y
