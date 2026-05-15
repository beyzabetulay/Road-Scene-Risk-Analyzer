"""
Road Scene Risk Analyzer — Image Analysis Pipeline

Orchestrates image loading → detection → result packaging.
This module is UI-independent and can be used from tests, CLI, or Streamlit.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

import numpy as np

from src.config import CONFIDENCE_THRESHOLD, YOLO_MODEL
from src.detection.detector import RoadDetector
from src.detection.schemas import Detection
from src.io.media_loader import ImageInput, load_image

logger = logging.getLogger(__name__)


# ── Result schema ───────────────────────────────────────────────


@dataclass
class AnalysisResult:
    """Container for the output of a single-image analysis run.

    Attributes:
        image_width:   Width of the analysed frame in pixels.
        image_height:  Height of the analysed frame in pixels.
        channels:      Number of colour channels (typically 3).
        detections:    List of :class:`Detection` objects found in the frame.
        detection_count: Number of detections (convenience field).
        timestamp:     ISO-8601 UTC timestamp of when the analysis was run.
        settings:      Dictionary of configuration values used for this run.
        source_name:   Optional human-readable name for the input source.
    """

    image_width: int
    image_height: int
    channels: int
    detections: list[Detection]
    detection_count: int
    timestamp: str
    settings: dict[str, Any]
    source_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire result to a JSON-friendly dictionary."""
        data = asdict(self)
        # asdict converts Detection tuples into lists — keep as-is for JSON.
        return data


# ── Pipeline function ───────────────────────────────────────────

# Module-level detector instance (lazy-loaded).
_detector: RoadDetector | None = None


def _get_detector(
    model_name: str = YOLO_MODEL,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> RoadDetector:
    """Return a shared detector, creating it on first call."""
    global _detector  # noqa: PLW0603
    if _detector is None:
        _detector = RoadDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
        )
    return _detector


def analyze_image(
    image_input: ImageInput,
    *,
    model_name: str = YOLO_MODEL,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    source_name: str = "",
) -> AnalysisResult:
    """Run the full image analysis pipeline.

    Steps:
        1. Load the image into a BGR NumPy array.
        2. Run YOLO detection through :class:`RoadDetector`.
        3. Package results into an :class:`AnalysisResult`.

    Args:
        image_input:          Anything accepted by :func:`load_image`
                              (path, bytes, file-like, or ndarray).
        model_name:           YOLO model to use (default from config).
        confidence_threshold: Minimum detection confidence (default from config).
        source_name:          Optional label for the input (e.g. filename).

    Returns:
        An :class:`AnalysisResult` containing image metadata, detections,
        timestamp, and the settings that were used.

    Raises:
        UnsupportedFormatError: If the file extension is not accepted.
        ImageLoadError: If decoding fails or the image is empty.
    """
    # 1. Load ─────────────────────────────────────────────────────
    image: np.ndarray = load_image(image_input)
    h, w, c = image.shape

    # Derive source name from path if not provided.
    if not source_name and isinstance(image_input, (str, Path)):
        source_name = Path(image_input).name

    logger.info(
        "Analyzing image '%s' (%dx%d, %d channels)",
        source_name or "<upload>",
        w,
        h,
        c,
    )

    # 2. Detect ───────────────────────────────────────────────────
    detector = _get_detector(model_name, confidence_threshold)
    detections: list[Detection] = detector.detect(image)

    # 3. Package ──────────────────────────────────────────────────
    result = AnalysisResult(
        image_width=w,
        image_height=h,
        channels=c,
        detections=detections,
        detection_count=len(detections),
        timestamp=datetime.now(timezone.utc).isoformat(),
        settings={
            "model_name": model_name,
            "confidence_threshold": confidence_threshold,
            "target_classes": list(detector.target_classes),
        },
        source_name=source_name,
    )

    logger.info(
        "Analysis complete: %d detection(s) found.",
        result.detection_count,
    )
    return result
