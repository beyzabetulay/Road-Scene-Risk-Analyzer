"""Tests for danger zone polygon creation and point-in-polygon detection.

Covers:
- Polygon shape and coordinate computation
- Point-in-polygon (inside, outside, edge cases)
- Default config parameters
- Non-square and high-resolution frames
- Asymmetric polygons

Refs: #13
"""

import numpy as np
import pytest

from src.risk.danger_zone import DangerZone, DangerZoneParams


# ── Polygon Creation Tests ──────────────────────────────────────


class TestPolygonCreation:
    """Tests for danger-zone polygon construction."""

    def test_polygon_shape_square_zone(self):
        """DangerZone computes a polygon of shape (4, 1, 2) with int32."""
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

    def test_polygon_absolute_coordinates(self):
        """Relative coords are correctly converted to pixel coordinates."""
        params = DangerZoneParams(
            top_left=(0.25, 0.25),
            top_right=(0.75, 0.25),
            bottom_right=(0.75, 0.75),
            bottom_left=(0.25, 0.75),
        )
        zone = DangerZone(100, 100, params)
        poly = zone.get_polygon()

        # Order: bottom_left, top_left, top_right, bottom_right
        expected = [
            [[25, 75]],  # bottom_left
            [[25, 25]],  # top_left
            [[75, 25]],  # top_right
            [[75, 75]],  # bottom_right
        ]
        np.testing.assert_array_equal(poly, expected)

    def test_polygon_rectangular_frame(self):
        """Polygon scales correctly on non-square (1920x1080) frames."""
        params = DangerZoneParams(
            top_left=(0.35, 0.50),
            top_right=(0.65, 0.50),
            bottom_right=(0.85, 0.95),
            bottom_left=(0.15, 0.95),
        )
        zone = DangerZone(1920, 1080, params)
        poly = zone.get_polygon()

        # bottom_left: (0.15 * 1920, 0.95 * 1080) = (288, 1026)
        assert poly[0, 0, 0] == int(0.15 * 1920)
        assert poly[0, 0, 1] == int(0.95 * 1080)

        # top_left: (0.35 * 1920, 0.50 * 1080) = (672, 540)
        assert poly[1, 0, 0] == int(0.35 * 1920)
        assert poly[1, 0, 1] == int(0.50 * 1080)

    def test_polygon_default_params(self):
        """DangerZone uses config defaults if no params are provided."""
        from src.config import DANGER_ZONE_BOTTOM_LEFT, DANGER_ZONE_TOP_LEFT

        zone = DangerZone(1280, 720)
        poly = zone.get_polygon()

        # bottom_left is first vertex
        expected_bl_x = int(DANGER_ZONE_BOTTOM_LEFT[0] * 1280)
        expected_bl_y = int(DANGER_ZONE_BOTTOM_LEFT[1] * 720)
        assert poly[0, 0, 0] == expected_bl_x
        assert poly[0, 0, 1] == expected_bl_y

        # top_left is second vertex
        expected_tl_x = int(DANGER_ZONE_TOP_LEFT[0] * 1280)
        expected_tl_y = int(DANGER_ZONE_TOP_LEFT[1] * 720)
        assert poly[1, 0, 0] == expected_tl_x
        assert poly[1, 0, 1] == expected_tl_y

    def test_polygon_full_frame(self):
        """A polygon covering the entire frame should be (0,0)→(W,H)."""
        params = DangerZoneParams(
            top_left=(0.0, 0.0),
            top_right=(1.0, 0.0),
            bottom_right=(1.0, 1.0),
            bottom_left=(0.0, 1.0),
        )
        zone = DangerZone(200, 100, params)
        poly = zone.get_polygon()

        # bottom_left=(0,100), top_left=(0,0), top_right=(200,0), bottom_right=(200,100)
        assert poly[0, 0, 0] == 0 and poly[0, 0, 1] == 100
        assert poly[1, 0, 0] == 0 and poly[1, 0, 1] == 0
        assert poly[2, 0, 0] == 200 and poly[2, 0, 1] == 0
        assert poly[3, 0, 0] == 200 and poly[3, 0, 1] == 100


# ── Point-in-Polygon Tests ─────────────────────────────────────


class TestPointInPolygon:
    """Tests for contains_point() method."""

    @pytest.fixture()
    def square_zone(self):
        """A simple square zone in the center of a 100x100 frame."""
        params = DangerZoneParams(
            top_left=(0.25, 0.25),
            top_right=(0.75, 0.25),
            bottom_right=(0.75, 0.75),
            bottom_left=(0.25, 0.75),
        )
        return DangerZone(100, 100, params)

    def test_point_inside_center(self, square_zone):
        """Center point is inside."""
        assert square_zone.contains_point((50, 50)) is True

    def test_point_outside_top_left_corner(self, square_zone):
        """Top-left corner of the frame is outside."""
        assert square_zone.contains_point((10, 10)) is False

    def test_point_outside_bottom_right_corner(self, square_zone):
        """Bottom-right corner of the frame is outside."""
        assert square_zone.contains_point((90, 90)) is False

    def test_point_on_left_edge(self, square_zone):
        """Point on the left edge of the polygon is considered inside."""
        assert square_zone.contains_point((25, 50)) is True

    def test_point_on_right_edge(self, square_zone):
        """Point on the right edge is considered inside."""
        assert square_zone.contains_point((75, 50)) is True

    def test_point_on_top_edge(self, square_zone):
        """Point on the top edge is considered inside."""
        assert square_zone.contains_point((50, 25)) is True

    def test_point_on_bottom_edge(self, square_zone):
        """Point on the bottom edge is considered inside."""
        assert square_zone.contains_point((50, 75)) is True

    def test_point_just_outside(self, square_zone):
        """Point 1px outside the polygon boundary."""
        assert square_zone.contains_point((24, 50)) is False

    def test_pedestrian_bottom_center_in_zone(self):
        """Simulate a pedestrian whose bottom-center is inside the danger zone.

        Acceptance criterion: A pedestrian inside the danger-zone is tested
        as HIGH risk — this test verifies the zone membership part.
        """
        # Trapezoid in lower-center of a 1280x720 frame
        params = DangerZoneParams(
            top_left=(0.35, 0.50),
            top_right=(0.65, 0.50),
            bottom_right=(0.85, 0.95),
            bottom_left=(0.15, 0.95),
        )
        zone = DangerZone(1280, 720, params)

        # Pedestrian standing at center bottom of the frame
        pedestrian_bottom_center = (640, 680)
        assert zone.contains_point(pedestrian_bottom_center) is True

    def test_distant_object_outside_zone(self):
        """Simulate a distant vehicle near the horizon, outside the zone.

        Acceptance criterion: A distant object outside danger-zone is tested
        as LOW or MEDIUM depending on formula.
        """
        params = DangerZoneParams(
            top_left=(0.35, 0.50),
            top_right=(0.65, 0.50),
            bottom_right=(0.85, 0.95),
            bottom_left=(0.15, 0.95),
        )
        zone = DangerZone(1280, 720, params)

        # Distant car near the horizon line (y=200, well above top of zone)
        distant_car_bottom_center = (640, 200)
        assert zone.contains_point(distant_car_bottom_center) is False
