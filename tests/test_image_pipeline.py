"""Tests for the image loading and analysis pipeline."""

from io import BytesIO

import cv2
import numpy as np
import pytest

from src.io.media_loader import (
    ImageLoadError,
    UnsupportedFormatError,
    load_image,
)


# ── Media Loader Tests ──────────────────────────────────────────


class TestLoadImageFromArray:
    """load_image() with NumPy array inputs."""

    def test_valid_bgr_array(self):
        """Accepts a valid (H, W, 3) array."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = load_image(img)
        assert result.shape == (100, 200, 3)

    def test_empty_array_raises(self):
        """Empty array raises ImageLoadError."""
        with pytest.raises(ImageLoadError, match="empty"):
            load_image(np.array([], dtype=np.uint8))

    def test_wrong_dimensions_raises(self):
        """2D grayscale array raises ImageLoadError."""
        with pytest.raises(ImageLoadError, match="3-channel"):
            load_image(np.zeros((100, 200), dtype=np.uint8))


class TestLoadImageFromBytes:
    """load_image() with raw bytes."""

    def test_valid_png_bytes(self):
        """Decodes a valid PNG from bytes."""
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        _, encoded = cv2.imencode(".png", img)
        result = load_image(encoded.tobytes())
        assert result.shape == (50, 50, 3)

    def test_empty_bytes_raises(self):
        """Empty bytes raises ImageLoadError."""
        with pytest.raises(ImageLoadError, match="empty"):
            load_image(b"")

    def test_corrupt_bytes_raises(self):
        """Random garbage bytes raises ImageLoadError."""
        with pytest.raises(ImageLoadError, match="corrupt"):
            load_image(b"not_an_image_at_all_random_garbage")


class TestLoadImageFromFileLike:
    """load_image() with file-like objects."""

    def test_bytesio_png(self):
        """Decodes PNG from a BytesIO object (simulates Streamlit upload)."""
        img = np.full((30, 40, 3), 200, dtype=np.uint8)
        _, encoded = cv2.imencode(".png", img)
        fobj = BytesIO(encoded.tobytes())
        result = load_image(fobj)
        assert result.shape == (30, 40, 3)


class TestLoadImageFromPath:
    """load_image() with file paths."""

    def test_nonexistent_path_raises(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image("/tmp/nonexistent_image_12345.jpg")

    def test_unsupported_extension_raises(self, tmp_path):
        """Unsupported extension raises UnsupportedFormatError."""
        bmp_file = tmp_path / "test.bmp"
        bmp_file.write_bytes(b"fake")
        with pytest.raises(UnsupportedFormatError, match="Unsupported"):
            load_image(str(bmp_file))

    def test_valid_jpg_from_disk(self, tmp_path):
        """Loads a valid JPEG from disk."""
        img = np.full((60, 80, 3), 100, dtype=np.uint8)
        path = tmp_path / "test.jpg"
        cv2.imwrite(str(path), img)
        result = load_image(str(path))
        assert result.shape[0] == 60
        assert result.shape[1] == 80

    def test_valid_png_from_disk(self, tmp_path):
        """Loads a valid PNG from disk."""
        img = np.full((40, 60, 3), 150, dtype=np.uint8)
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), img)
        result = load_image(str(path))
        assert result.shape[:2] == (40, 60)


class TestLoadImageTypeError:
    """load_image() with unsupported types."""

    def test_int_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            load_image(42)  # type: ignore

    def test_list_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            load_image([1, 2, 3])  # type: ignore


# ── Pipeline Tests ──────────────────────────────────────────────


def _model_available() -> bool:
    """Check if YOLO model weights can be loaded."""
    try:
        from ultralytics import YOLO
        YOLO("yolov8n.pt")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _model_available(), reason="YOLO weights not available")
class TestAnalyzeImage:
    """Integration tests for analyze_image (require model weights)."""

    def test_analyze_numpy_array(self):
        """analyze_image accepts a numpy array and returns AnalysisResult."""
        from src.pipeline import analyze_image

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = analyze_image(img, source_name="test_blank")
        assert result.image_width == 640
        assert result.image_height == 480
        assert result.channels == 3
        assert isinstance(result.detections, list)
        assert result.detection_count == len(result.detections)
        assert result.timestamp  # non-empty
        assert result.source_name == "test_blank"

    def test_result_to_dict(self):
        """AnalysisResult.to_dict() returns a JSON-serializable dict."""
        import json
        from src.pipeline import analyze_image

        img = np.zeros((240, 320, 3), dtype=np.uint8)
        result = analyze_image(img)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "detections" in d
        assert "settings" in d
        # Should be JSON-serializable without error.
        json.dumps(d)

    def test_settings_captured(self):
        """Settings dict contains model and threshold info."""
        from src.pipeline import analyze_image

        img = np.zeros((240, 320, 3), dtype=np.uint8)
        result = analyze_image(img)
        assert "model_name" in result.settings
        assert "confidence_threshold" in result.settings
        assert "target_classes" in result.settings

    def test_analyze_from_bytes(self):
        """analyze_image accepts PNG bytes."""
        from src.pipeline import analyze_image

        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        _, encoded = cv2.imencode(".png", img)
        result = analyze_image(encoded.tobytes())
        assert result.image_width == 100
        assert result.image_height == 100

    def test_invalid_input_raises(self):
        """Invalid input type raises TypeError."""
        from src.pipeline import analyze_image

        with pytest.raises(TypeError):
            analyze_image(12345)  # type: ignore
