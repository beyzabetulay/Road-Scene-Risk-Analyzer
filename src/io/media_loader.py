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

from src.config import IMAGE_EXTENSIONS

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
