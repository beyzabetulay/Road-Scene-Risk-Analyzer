"""Tests for video loading and analysis pipeline."""

import cv2
import numpy as np
import pytest

from src.io.media_loader import (
    UnsupportedFormatError,
    VideoInfo,
    VideoLoadError,
    get_video_info,
    load_video_frames,
)
from src.pipeline import analyze_video

# ── Helpers to create dummy videos ──────────────────────────────


def create_dummy_video(path: str, frames: int = 30, fps: float = 30.0) -> None:
    """Create a short dummy video file for testing."""
    height, width = 240, 320
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(frames):
        # Create a frame with a moving grey square to simulate motion
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x = (i * 5) % width
        cv2.rectangle(frame, (x, 50), (x + 50, 100), (128, 128, 128), -1)
        out.write(frame)
    out.release()


# ── Media Loader (Video) Tests ──────────────────────────────────


def test_get_video_info(tmp_path):
    """get_video_info extracts correct metadata."""
    vid_path = tmp_path / "test.mp4"
    create_dummy_video(str(vid_path), frames=15, fps=15.0)

    info = get_video_info(vid_path)
    assert isinstance(info, VideoInfo)
    assert info.total_frames == 15
    assert info.fps == 15.0
    assert info.width == 320
    assert info.height == 240
    assert info.duration_s == 1.0


def test_get_video_info_not_found():
    """Missing video raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        get_video_info("/tmp/does_not_exist_123.mp4")


def test_get_video_info_unsupported(tmp_path):
    """Unsupported extension raises UnsupportedFormatError."""
    vid_path = tmp_path / "test.mkv"
    vid_path.write_bytes(b"dummy")
    with pytest.raises(UnsupportedFormatError):
        get_video_info(vid_path)


def test_load_video_frames_stride(tmp_path):
    """load_video_frames respects stride and yields (idx, frame)."""
    vid_path = tmp_path / "test.mp4"
    create_dummy_video(str(vid_path), frames=20)

    frames = list(load_video_frames(vid_path, stride=5))
    # Should get frames 0, 5, 10, 15
    assert len(frames) == 4
    for idx, (frame_idx, frame) in enumerate(frames):
        assert frame_idx == idx * 5
        assert frame.shape == (240, 320, 3)


# ── Pipeline (Video) Tests ──────────────────────────────────────


def _model_available() -> bool:
    try:
        from ultralytics import YOLO
        YOLO("yolov8n.pt")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _model_available(), reason="YOLO weights not available")
class TestAnalyzeVideo:
    """Integration tests for analyze_video (require model weights)."""

    def test_analyze_video_integration(self, tmp_path):
        """analyze_video processes a video and returns aggregate stats."""
        vid_path = tmp_path / "test.mp4"
        # 12 frames total, stride 4 -> should process frames 0, 4, 8 (3 frames)
        create_dummy_video(str(vid_path), frames=12)

        result = analyze_video(vid_path, stride=4)

        assert result.video_info.total_frames == 12
        assert result.frame_stride == 4
        assert result.frames_processed == 3
        # Since it's a dummy video with no cars/people, detections should be 0
        assert result.avg_object_count == 0.0
        assert result.high_risk_frames == 0
        assert result.max_scene_risk == 0.0
        assert len(result.frame_results) == 3

    def test_video_result_to_dict(self, tmp_path):
        """VideoAnalysisResult serializes cleanly to dict."""
        import json
        vid_path = tmp_path / "test.mp4"
        create_dummy_video(str(vid_path), frames=5)

        result = analyze_video(vid_path, stride=5)
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "frame_results" in d
        assert "video_info" in d
        # Verify JSON serialization works without error
        json.dumps(d)
