"""
Risk — Danger Zone Mapping

Generates polygon-based danger zone overlays on frames
and checks whether objects lie within the zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from src.config import (
    DANGER_ZONE_BOTTOM_LEFT,
    DANGER_ZONE_BOTTOM_RIGHT,
    DANGER_ZONE_TOP_LEFT,
    DANGER_ZONE_TOP_RIGHT,
)


@dataclass(frozen=True)
class DangerZoneParams:
    """Relative coordinates for the danger zone polygon.

    Coordinates are expressed as fractions of the frame dimensions
    (x_ratio, y_ratio), where (0, 0) is top-left and (1, 1) is bottom-right.
    """
    top_left: Tuple[float, float] = DANGER_ZONE_TOP_LEFT
    top_right: Tuple[float, float] = DANGER_ZONE_TOP_RIGHT
    bottom_right: Tuple[float, float] = DANGER_ZONE_BOTTOM_RIGHT
    bottom_left: Tuple[float, float] = DANGER_ZONE_BOTTOM_LEFT


class DangerZone:
    """Represents a danger zone polygon for a specific frame size.

    Computes the absolute pixel coordinates of the polygon based on
    the relative parameters and provides methods for point-in-polygon
    testing.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        params: DangerZoneParams | None = None,
    ) -> None:
        """Initialize the danger zone for a specific resolution.

        Args:
            frame_width:  Width of the image in pixels.
            frame_height: Height of the image in pixels.
            params:       Optional relative coordinates. Defaults to config values.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.params = params or DangerZoneParams()
        self.polygon = self._compute_polygon()

    def _compute_polygon(self) -> np.ndarray:
        """Convert relative parameter coordinates to absolute pixel coordinates."""
        pts = [
            self.params.bottom_left,
            self.params.top_left,
            self.params.top_right,
            self.params.bottom_right,
        ]

        # Convert to absolute integer coordinates
        abs_pts = [
            [int(x * self.frame_width), int(y * self.frame_height)]
            for x, y in pts
        ]

        # OpenCV expects an array of shape (N, 1, 2) with dtype int32
        return np.array(abs_pts, dtype=np.int32).reshape((-1, 1, 2))

    def get_polygon(self) -> np.ndarray:
        """Return the computed polygon array for drawing."""
        return self.polygon

    def contains_point(self, point: tuple[int, int]) -> bool:
        """Check if a point lies inside the danger zone.

        Args:
            point: (x, y) pixel coordinates.

        Returns:
            True if the point is strictly inside or on the edge of the polygon.
        """
        # cv2.pointPolygonTest returns:
        #  > 0 if point is inside
        #  = 0 if point is on the edge
        #  < 0 if point is outside
        dist = cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), measureDist=False)
        return dist >= 0
