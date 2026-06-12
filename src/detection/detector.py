"""
Detection — YOLO-based Object Detector

Wraps Ultralytics YOLOv8 for road scene object detection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from ultralytics import YOLO

from src.config import CONFIDENCE_THRESHOLD, TARGET_CLASSES, YOLO_MODEL
from src.detection.schemas import Detection

if TYPE_CHECKING:  # pragma: no cover
    from ultralytics.engine.results import Results

logger = logging.getLogger(__name__)


class RoadDetector:
    """YOLO-based object detector filtered for road-relevant classes.

    The model is loaded once at instantiation and reused for all
    subsequent calls. Only detections whose class name appears in
    *target_classes* and whose confidence meets the threshold are
    returned.

    Parameters:
        model_name:           Ultralytics model identifier or path.
        confidence_threshold: Minimum confidence to keep a detection.
        target_classes:       Allowlist of COCO class names to retain.
    """

    def __init__(
        self,
        model_name: str = YOLO_MODEL,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        target_classes: list[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.target_classes = set(target_classes or TARGET_CLASSES)

        logger.info("Loading YOLO model: %s", self.model_name)
        self.model = YOLO(self.model_name)
        logger.info("Model loaded successfully.")

    # ── public API ──────────────────────────────────────────────

    def detect(self, image: np.ndarray, persist: bool = False) -> list[Detection]:
        """Run inference on a single image and return filtered detections.

        Args:
            image: BGR NumPy array (H, W, 3) as returned by OpenCV.
            persist: Whether to use object tracking across frames (ByteTrack).

        Returns:
            List of :class:`Detection` objects for road-relevant classes
            that meet the confidence threshold. The list may be empty.
        """
        if image is None or image.size == 0:
            logger.warning("Received empty image, returning no detections.")
            return []

        frame_h, frame_w = image.shape[:2]
        frame_area = frame_h * frame_w

        if persist:
            results: list[Results] = self.model.track(
                image,
                conf=self.confidence_threshold,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
        else:
            results: list[Results] = self.model(
                image,
                conf=self.confidence_threshold,
                verbose=False,
            )

        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            detections.extend(
                self._parse_boxes(result, frame_w, frame_h, frame_area)
            )

        logger.debug(
            "Detected %d road-relevant objects (out of %d total).",
            len(detections),
            sum(len(r.boxes) for r in results if r.boxes is not None),
        )
        return detections

    # ── internals ───────────────────────────────────────────────

    def _parse_boxes(
        self,
        result: Results,
        frame_w: int,
        frame_h: int,
        frame_area: int,
    ) -> list[Detection]:
        """Convert a single YOLO Results object into Detection instances."""
        detections: list[Detection] = []
        names = result.names  # {int: str} mapping from the model

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = names.get(class_id, f"class_{class_id}")

            if class_name not in self.target_classes:
                continue

            confidence = float(box.conf[0])
            
            # Extract track ID if available
            track_id = int(box.id[0]) if box.id is not None else None

            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            area_ratio = (bbox_w * bbox_h) / frame_area if frame_area else 0.0

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=round(confidence, 4),
                    bbox_xyxy=(x1, y1, x2, y2),
                    bbox_width=bbox_w,
                    bbox_height=bbox_h,
                    bbox_area_ratio=round(area_ratio, 6),
                    center=(cx, cy),
                    bottom_center=(cx, y2),
                    track_id=track_id,
                )
            )

        return detections
