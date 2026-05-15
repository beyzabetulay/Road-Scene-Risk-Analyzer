"""
I/O — Media Loader

Utilities for loading images and video frames from disk or uploads.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Union

import cv2
import numpy as np
from PIL import Image

from src.config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, VIDEO_FRAME_STRIDE

logger = logging.getLogger(__name__)

# Type alias for inputs accepted by the loader.
ImageInput = Union[str, Path, bytes, BinaryIO, np.ndarray]


class UnsupportedFormatError(Exception):
    """Raised when the file extension is not in the accepted set."""


class ImageLoadError(Exception):
    """Raised when the image cannot be decoded."""


def load_image(source: ImageInput) -> np.ndarray:
    """Load an image from various sources and return a BGR NumPy array.

    Supported sources:
        - File path (``str`` or ``Path``) — must have an accepted extension.
        - Raw bytes (``bytes``) — decoded via OpenCV.
        - File-like object (``BinaryIO``) — e.g. Streamlit ``UploadedFile``.
        - NumPy array — returned as-is after basic validation.

    Args:
        source: The image to load.

    Returns:
        BGR ``np.ndarray`` with shape ``(H, W, 3)``.

    Raises:
        UnsupportedFormatError: If the file extension is not accepted.
        ImageLoadError: If decoding fails or the image is empty.
        TypeError: If *source* is an unsupported type.
    """
    if isinstance(source, np.ndarray):
        return _validate_array(source)

    if isinstance(source, (str, Path)):
        return _load_from_path(Path(source))

    if isinstance(source, bytes):
        return _decode_bytes(source)

    # File-like object (e.g. Streamlit UploadedFile, open() handle).
    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, str):
            raise ImageLoadError("File-like object returned str, expected bytes.")
        return _decode_bytes(raw)

    raise TypeError(
        f"Unsupported image source type: {type(source).__name__}. "
        "Expected str, Path, bytes, file-like, or np.ndarray."
    )


# ── Internal helpers ────────────────────────────────────────────


def _load_from_path(path: Path) -> np.ndarray:
    """Load an image file from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported image format '{suffix}'. "
            f"Accepted: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError(f"Failed to decode image: {path}")

    logger.info("Loaded image from %s (%dx%d)", path.name, image.shape[1], image.shape[0])
    return image


def _decode_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR NumPy array."""
    if not data:
        raise ImageLoadError("Received empty bytes — cannot decode image.")

    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError("Failed to decode image from bytes (corrupt or unsupported).")

    logger.info("Decoded image from bytes (%dx%d)", image.shape[1], image.shape[0])
    return image


def _validate_array(arr: np.ndarray) -> np.ndarray:
    """Validate that a NumPy array looks like a BGR image."""
    if arr.size == 0:
        raise ImageLoadError("Received empty NumPy array.")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ImageLoadError(
            f"Expected a 3-channel image (H, W, 3), got shape {arr.shape}."
        )
    return arr


# ── Video loading ───────────────────────────────────────────────

from dataclasses import dataclass
from typing import Generator


class VideoLoadError(Exception):
    """Raised when a video file cannot be opened or read."""


@dataclass(frozen=True)
class VideoInfo:
    """Metadata extracted from a video file before frame iteration.

    Attributes:
        path:         Absolute path to the video file.
        total_frames: Total number of frames reported by the codec.
        fps:          Frames per second.
        width:        Frame width in pixels.
        height:       Frame height in pixels.
        duration_s:   Approximate duration in seconds.
    """

    path: str
    total_frames: int
    fps: float
    width: int
    height: int
    duration_s: float


def get_video_info(video_path: str | Path) -> VideoInfo:
    """Open a video file and return its metadata without reading frames.

    Args:
        video_path: Path to an ``.mp4`` or ``.avi`` file.

    Returns:
        A :class:`VideoInfo` with codec-reported metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        UnsupportedFormatError: If the extension is not accepted.
        VideoLoadError: If OpenCV cannot open the file.
    """
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported video format '{suffix}'. "
            f"Accepted: {', '.join(sorted(VIDEO_EXTENSIONS))}"
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoLoadError(f"Failed to open video: {path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = round(total_frames / fps, 2) if fps > 0 else 0.0
    finally:
        cap.release()

    info = VideoInfo(
        path=str(path.resolve()),
        total_frames=total_frames,
        fps=fps,
        width=width,
        height=height,
        duration_s=duration_s,
    )
    logger.info(
        "Video info: %s — %d frames, %.1f fps, %dx%d, %.1fs",
        path.name,
        total_frames,
        fps,
        width,
        height,
        duration_s,
    )
    return info


def load_video_frames(
    video_path: str | Path,
    *,
    stride: int = VIDEO_FRAME_STRIDE,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Yield ``(frame_index, bgr_array)`` tuples from a video file.

    Only every *stride*-th frame is yielded to keep processing time
    manageable.  Frames are read one at a time so the full video is
    **never** loaded into memory.

    Args:
        video_path: Path to an ``.mp4`` or ``.avi`` file.
        stride:     Process every *stride*-th frame (default from config).

    Yields:
        ``(frame_index, frame)`` where *frame* is a BGR ``np.ndarray``.

    Raises:
        FileNotFoundError: If the file does not exist.
        UnsupportedFormatError: If the extension is not accepted.
        VideoLoadError: If OpenCV cannot open the file.
    """
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported video format '{suffix}'. "
            f"Accepted: {', '.join(sorted(VIDEO_EXTENSIONS))}"
        )

    stride = max(1, stride)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoLoadError(f"Failed to open video: {path}")

    frame_idx = 0
    try:
        while True:
            grabbed = cap.grab()
            if not grabbed:
                break

            if frame_idx % stride == 0:
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    yield frame_idx, frame

            frame_idx += 1
    finally:
        cap.release()

    logger.info(
        "Finished reading video: %d frames iterated, stride=%d",
        frame_idx,
        stride,
    )
