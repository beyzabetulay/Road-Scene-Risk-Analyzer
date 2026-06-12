"""
Road Scene Risk Analyzer — Analysis Pipeline

Orchestrates image/video loading → detection → result packaging.
This module is UI-independent and can be used from tests, CLI, or Streamlit.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

import numpy as np

from src.config import (
    CONFIDENCE_THRESHOLD,
    RISK_THRESHOLD_HIGH,
    VIDEO_FRAME_STRIDE,
    YOLO_MODEL,
)
from src.visualization.annotator import annotate_video_frame
from src.io.video_writer import AnnotatedVideoWriter
from src.detection.detector import RoadDetector
from src.detection.schemas import Detection
from src.io.media_loader import (
    ImageInput,
    VideoInfo,
    get_video_info,
    load_image,
    load_video_frames,
)

from src.risk.scene_classifier import SceneRiskResult, classify_scene
from src.risk.scoring import calculate_object_risk, get_risk_reason

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
        scene_risk:    Overall scene risk classification and summary.
        timestamp:     ISO-8601 UTC timestamp of when the analysis was run.
        settings:      Dictionary of configuration values used for this run.
        source_name:   Optional human-readable name for the input source.
    """

    image_width: int
    image_height: int
    channels: int
    detections: list[Detection]
    detection_count: int
    scene_risk: SceneRiskResult
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
    danger_zone_params: DangerZoneParams | None = None,
    use_depth: bool = False,
) -> AnalysisResult:
    """Run detection on a single image and return results.

    Args:
        image_input:          The image to process
                              (path, bytes, file-like, or ndarray).
        model_name:           YOLO model to use (default from config).
        confidence_threshold: Minimum detection confidence (default from config).
        source_name:          Optional label for the input (e.g. filename).
        danger_zone_params:   Optional danger zone parameters.
        use_depth:            Whether to estimate depth for detections using MiDaS.

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

    # Risk processing: Danger Zone ────────────────────────────────
    from src.risk.danger_zone import DangerZone, DangerZoneParams
    zone = DangerZone(w, h, params=danger_zone_params)
    
    # Estimate Depth if enabled ───────────────────────────────────
    depth_map = None
    if use_depth:
        from src.depth.estimator import DepthEstimator
        from src.config import DEPTH_MODEL
        depth_estimator = DepthEstimator(model_type=DEPTH_MODEL)
        depth_map = depth_estimator.estimate(image)
    
    updated_detections = []
    for d in detections:
        is_in_zone = zone.contains_point(d.bottom_center)
        
        est_depth = 0.0
        if depth_map is not None:
            est_depth = DepthEstimator.get_depth_at_point(depth_map, d.center)
            
        temp_d = d.with_risk(in_danger_zone=is_in_zone, estimated_depth=est_depth)
        score = calculate_object_risk(temp_d, h)
        reason = get_risk_reason(temp_d)
        updated_detections.append(
            temp_d.with_risk(risk_score=score, risk_reason=reason)
        )
        
    scene_risk = classify_scene(updated_detections)
    
    # 3. Package ──────────────────────────────────────────────────
    result = AnalysisResult(
        image_width=w,
        image_height=h,
        channels=c,
        detections=updated_detections,
        detection_count=len(updated_detections),
        scene_risk=scene_risk,
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


# ── Video analysis ──────────────────────────────────────────────


@dataclass
class VideoFrameResult:
    """Detection result for a single processed video frame.

    Attributes:
        frame_index:     Original frame index in the video.
        detections:      List of Detection objects for this frame.
        detection_count: Number of detections in this frame.
        max_risk_score:  Highest object-level risk score (0 before risk scoring).
    """

    frame_index: int
    detections: list[Detection]
    detection_count: int
    max_risk_score: float
    scene_risk_level: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return asdict(self)


@dataclass
class VideoAnalysisResult:
    """Aggregate result of analysing a video file.

    Attributes:
        video_info:          Metadata about the source video.
        frame_stride:        How many frames were skipped between processed frames.
        frame_results:       Per-frame detection results.
        total_frames_read:   Total frames iterated (including skipped).
        frames_processed:    Number of frames actually analysed.
        max_scene_risk:      Highest per-frame max risk score across all frames.
        avg_object_count:    Average number of detections per processed frame.
        high_risk_frames:    Number of frames whose max risk score >= HIGH threshold.
        riskiest_frame_index: Frame index with the highest max risk score (-1 if none).
        timestamp:           ISO-8601 UTC timestamp of when the analysis completed.
        settings:            Configuration values used for this run.
        annotated_video_path: Optional path to the generated annotated video file.
    """

    video_info: VideoInfo
    frame_stride: int
    frame_results: list[VideoFrameResult]
    total_frames_read: int
    frames_processed: int
    max_scene_risk: float
    avg_object_count: float
    high_risk_frames: int
    riskiest_frame_index: int
    timestamp: str
    settings: dict[str, Any]
    annotated_video_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire result to a JSON-friendly dictionary."""
        return asdict(self)


def analyze_video(
    video_path: str | Path,
    *,
    stride: int = VIDEO_FRAME_STRIDE,
    model_name: str = YOLO_MODEL,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    danger_zone_params: DangerZoneParams | None = None,
    output_video_path: str | Path | None = None,
    use_lane_detection: bool = False,
    use_depth: bool = False,
) -> VideoAnalysisResult:
    """Run detection on every *stride*-th frame of a video.

    Args:
        video_path:           Path to ``.mp4`` or ``.avi`` file.
        stride:               Process every N-th frame.
        model_name:           YOLO model to use.
        confidence_threshold: Minimum detection confidence.
        danger_zone_params:   Optional danger zone parameters.
        output_video_path:    Optional path to write an annotated MP4 video.
        use_lane_detection:   Whether to use dynamic lane detection for the danger zone.
        use_depth:            Whether to estimate depth for detections using MiDaS.

    Returns:
        A :class:`VideoAnalysisResult` with per-frame results and
        aggregate summary statistics.
    """
    video_path = Path(video_path)
    info = get_video_info(video_path)

    logger.info(
        "Starting video analysis: %s (stride=%d)",
        video_path.name,
        stride,
    )

    detector = _get_detector(model_name, confidence_threshold)
    # Initialize DangerZone for the video resolution
    from src.risk.danger_zone import DangerZone, DangerZoneParams
    static_zone = DangerZone(info.width, info.height, params=danger_zone_params)
    
    # Initialize LaneDetector if enabled
    lane_detector = None
    if use_lane_detection:
        from src.risk.lane_detector import LaneDetector
        from src.config import (
            LANE_CANNY_LOW, LANE_CANNY_HIGH,
            LANE_HOUGH_THRESHOLD, LANE_MIN_LINE_LENGTH, LANE_MAX_LINE_GAP
        )
        lane_detector = LaneDetector(
            canny_low=LANE_CANNY_LOW,
            canny_high=LANE_CANNY_HIGH,
            hough_threshold=LANE_HOUGH_THRESHOLD,
            min_line_length=LANE_MIN_LINE_LENGTH,
            max_line_gap=LANE_MAX_LINE_GAP,
        )

    # Initialize DepthEstimator if enabled
    depth_estimator = None
    if use_depth:
        from src.depth.estimator import DepthEstimator
        from src.config import DEPTH_MODEL
        depth_estimator = DepthEstimator(model_type=DEPTH_MODEL)

    frame_results: list[VideoFrameResult] = []
    total_read = 0
    
    writer: AnnotatedVideoWriter | None = None
    if output_video_path is not None:
        effective_fps = info.fps / stride if stride > 1 else info.fps
        writer = AnnotatedVideoWriter(output_video_path, effective_fps, info.width, info.height)
        writer.open()

    try:
        for frame_idx, frame in load_video_frames(video_path, stride=stride):
            total_read = frame_idx + 1
            detections = detector.detect(frame)
            # Estimate Depth if enabled
            depth_map = None
            if depth_estimator is not None:
                depth_map = depth_estimator.estimate(frame)
            
            # Update Danger Zone dynamically if enabled
            current_zone = static_zone
            if lane_detector is not None:
                poly = lane_detector.get_dynamic_danger_zone(frame)
                if poly is not None:
                    current_zone = DangerZone.from_lane_polygon(poly, info.width, info.height)
            
            # Risk processing: Danger Zone and Scoring
            updated_detections = []
            for d in detections:
                is_in_zone = current_zone.contains_point(d.bottom_center)
                
                est_depth = 0.0
                if depth_map is not None:
                    # Use center point for depth lookup
                    est_depth = DepthEstimator.get_depth_at_point(depth_map, d.center)
                    
                temp_d = d.with_risk(in_danger_zone=is_in_zone, estimated_depth=est_depth)
                score = calculate_object_risk(temp_d, info.height)
                reason = get_risk_reason(temp_d)
                updated_detections.append(
                    temp_d.with_risk(risk_score=score, risk_reason=reason)
                )

            scene_risk = classify_scene(updated_detections)

            frame_results.append(
                VideoFrameResult(
                    frame_index=frame_idx,
                    detections=updated_detections,
                    detection_count=len(updated_detections),
                    max_risk_score=scene_risk.max_risk_score,
                    scene_risk_level=scene_risk.risk_level,
                )
            )
            
            if writer is not None:
                ann_frame = annotate_video_frame(
                    frame,
                    updated_detections,
                    scene_risk,
                    danger_zone=current_zone,
                )
                writer.write_frame(ann_frame)
    finally:
        if writer is not None:
            writer.close()

    # ── Aggregate summary ────────────────────────────────────────
    frames_processed = len(frame_results)

    if frames_processed > 0:
        max_scene_risk = max(fr.max_risk_score for fr in frame_results)
        avg_object_count = round(
            sum(fr.detection_count for fr in frame_results) / frames_processed,
            2,
        )
        high_risk_frames = sum(
            1 for fr in frame_results
            if fr.max_risk_score >= RISK_THRESHOLD_HIGH
        )
        riskiest_frame_index = max(
            frame_results, key=lambda fr: fr.max_risk_score
        ).frame_index
    else:
        max_scene_risk = 0.0
        avg_object_count = 0.0
        high_risk_frames = 0
        riskiest_frame_index = -1

    result = VideoAnalysisResult(
        video_info=info,
        frame_stride=stride,
        frame_results=frame_results,
        total_frames_read=total_read,
        frames_processed=frames_processed,
        max_scene_risk=max_scene_risk,
        avg_object_count=avg_object_count,
        high_risk_frames=high_risk_frames,
        riskiest_frame_index=riskiest_frame_index,
        timestamp=datetime.now(timezone.utc).isoformat(),
        settings={
            "model_name": model_name,
            "confidence_threshold": confidence_threshold,
            "frame_stride": stride,
            "target_classes": list(detector.target_classes),
        },
        annotated_video_path=str(output_video_path) if output_video_path else None,
    )

    logger.info(
        "Video analysis complete: %d/%d frames processed, "
        "max_risk=%.1f, avg_objects=%.1f, high_risk_frames=%d",
        frames_processed,
        total_read,
        max_scene_risk,
        avg_object_count,
        high_risk_frames,
    )
    return result
