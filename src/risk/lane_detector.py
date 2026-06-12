"""
Risk Analysis — Lane Detector

Dynamically detects lane lines in a video frame using traditional computer vision
(Canny Edge Detection + Probabilistic Hough Transform) and converts them into
a dynamic Danger Zone polygon.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LaneDetector:
    """Detects lane lines and constructs a dynamic danger zone."""

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 50,
        min_line_length: int = 100,
        max_line_gap: int = 50,
    ) -> None:
        """Initialize the lane detector.

        Args:
            canny_low: Lower threshold for Canny edge detection.
            canny_high: Upper threshold for Canny edge detection.
            hough_threshold: Minimum number of intersections to detect a line.
            min_line_length: Minimum length of a line segment.
            max_line_gap: Maximum allowed gap between line segments.
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    def detect_lanes(
        self, frame: np.ndarray
    ) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
        """Detect left and right lane lines in the given frame.

        Args:
            frame: Original BGR image as a NumPy array.

        Returns:
            A tuple of (left_line, right_line), where each line is represented as
            (x1, y1, x2, y2). Returns None for a line if not detected.
        """
        height, width = frame.shape[:2]

        # 1. Grayscale & Blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Edge Detection
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        # 3. ROI Masking (lower half of the image, trapezoid shape)
        mask = np.zeros_like(edges)
        polygon = np.array(
            [
                [
                    (int(width * 0.1), height),
                    (int(width * 0.45), int(height * 0.6)),
                    (int(width * 0.55), int(height * 0.6)),
                    (int(width * 0.9), height),
                ]
            ],
            np.int32,
        )
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # 4. Hough Transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        if lines is None:
            return None, None

        # 5. Separate into Left and Right lines
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue  # Prevent division by zero
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter out near-horizontal lines
            if abs(slope) < 0.3:
                continue

            if slope < 0:
                # Left lane (negative slope, points going up-right in image coords)
                if x1 < width * 0.6 and x2 < width * 0.6:
                    left_lines.append((x1, y1, x2, y2))
            else:
                # Right lane (positive slope, points going down-right in image coords)
                if x1 > width * 0.4 and x2 > width * 0.4:
                    right_lines.append((x1, y1, x2, y2))

        # 6. Average lines
        left_lane = self._average_lines(left_lines, height) if left_lines else None
        right_lane = self._average_lines(right_lines, height) if right_lines else None

        return left_lane, right_lane

    def _average_lines(self, lines: list[tuple[int, int, int, int]], image_height: int) -> tuple[int, int, int, int] | None:
        """Average a list of line segments and extrapolate to the ROI boundaries."""
        if not lines:
            return None

        x_coords = []
        y_coords = []

        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        if len(x_coords) < 2:
            return None

        # Fit a linear polynomial (y = mx + b)
        try:
            poly = np.polyfit(y_coords, x_coords, deg=1)
            # Extrapolate to bottom of image and ROI top (60% of height)
            y1 = image_height
            y2 = int(image_height * 0.6)
            x1 = int(np.polyval(poly, y1))
            x2 = int(np.polyval(poly, y2))
            return x1, y1, x2, y2
        except Exception as e:
            logger.debug("Failed to average lines: %s", e)
            return None

    def get_dynamic_danger_zone(
        self, frame: np.ndarray
    ) -> np.ndarray | None:
        """Construct a danger zone polygon from detected lanes.

        Returns:
            A NumPy array of shape (4, 2) representing the polygon vertices,
            or None if both lanes could not be detected reliably.
        """
        left_lane, right_lane = self.detect_lanes(frame)

        if left_lane is None or right_lane is None:
            return None

        # Unpack lines
        lx1, ly1, lx2, ly2 = left_lane
        rx1, ry1, rx2, ry2 = right_lane

        # Construct polygon: bottom-left, top-left, top-right, bottom-right
        polygon = np.array(
            [
                [lx1, ly1],
                [lx2, ly2],
                [rx2, ry2],
                [rx1, ry1],
            ],
            np.int32,
        )

        return polygon
