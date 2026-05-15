"""
Result Exporters

Provides functions to format and export analysis results 
into JSON, CSV, and image byte streams for UI downloads.
"""

import csv
import io
import json
from datetime import datetime, timezone
from typing import Union

import cv2
import numpy as np

from src.pipeline import AnalysisResult, VideoAnalysisResult


def generate_export_filename(prefix: str, risk_level: str, ext: str) -> str:
    """Generate a consistent filename with timestamp and risk level.

    Args:
        prefix: Prefix for the filename (e.g., 'image_report').
        risk_level: Risk level string (LOW, MEDIUM, HIGH).
        ext: File extension including dot (e.g., '.json').
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{risk_level}_{ts}{ext}"


def export_report_to_json(result: Union[AnalysisResult, VideoAnalysisResult]) -> str:
    """Export the entire analysis result as a formatted JSON string."""
    data = result.to_dict()
    return json.dumps(data, indent=2)


def export_table_to_csv(result: Union[AnalysisResult, VideoAnalysisResult]) -> str:
    """Export detections to a CSV string.

    For images, exports all detections.
    For videos, exports all detections across all processed frames.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    if isinstance(result, AnalysisResult):
        writer.writerow(
            ["Class", "Confidence", "Risk Score", "In Danger Zone", "Reason"]
        )
        for d in result.detections:
            writer.writerow(
                [
                    d.class_name,
                    f"{d.confidence:.4f}",
                    d.risk_score,
                    d.in_danger_zone,
                    d.risk_reason,
                ]
            )
    else:
        writer.writerow(
            [
                "Frame Index",
                "Class",
                "Confidence",
                "Risk Score",
                "In Danger Zone",
                "Reason",
            ]
        )
        for frame_res in result.frame_results:
            for d in frame_res.detections:
                writer.writerow(
                    [
                        frame_res.frame_index,
                        d.class_name,
                        f"{d.confidence:.4f}",
                        d.risk_score,
                        d.in_danger_zone,
                        d.risk_reason,
                    ]
                )

    return output.getvalue()


def export_image_to_bytes(image: np.ndarray, ext: str = ".png") -> bytes:
    """Encode an OpenCV BGR image array into bytes for downloading."""
    success, encoded_image = cv2.imencode(ext, image)
    if not success:
        raise RuntimeError(f"Failed to encode image to {ext}")
    return encoded_image.tobytes()
