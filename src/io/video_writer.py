"""
I/O — Video Writer

Utilities for writing annotated video frames to an output video file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AnnotatedVideoWriter:
    """Writes BGR numpy arrays (frames) sequentially to an MP4 video file.

    Designed to be used as a context manager:
        with AnnotatedVideoWriter("out.mp4", 30.0, 1920, 1080) as writer:
            writer.write_frame(frame1)
            writer.write_frame(frame2)
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
    ) -> None:
        """Initialize the video writer.

        Args:
            output_path: Path where the output video will be saved.
            fps: Frames per second of the output video.
            width: Width of the video frames.
            height: Height of the video frames.
            codec: FourCC codec string (default: "mp4v").
        """
        self.output_path = Path(output_path)
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.codec = codec

        self._writer: cv2.VideoWriter | None = None
        self._frames_written = 0

    def open(self) -> AnnotatedVideoWriter:
        """Open the video writer resource."""
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height),
            isColor=True,
        )

        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {self.output_path}")

        logger.info(
            "Opened video writer: %s (%dx%d at %.1f fps)",
            self.output_path.name,
            self.width,
            self.height,
            self.fps,
        )
        return self

    def close(self) -> None:
        """Release the video writer resource."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info(
                "Closed video writer: %s (%d frames written)",
                self.output_path.name,
                self._frames_written,
            )

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single BGR frame to the video.

        Args:
            frame: A BGR numpy array of shape (height, width, 3).

        Raises:
            RuntimeError: If the writer is not open.
            ValueError: If the frame dimensions do not match the writer's dimensions.
        """
        if self._writer is None:
            raise RuntimeError("Cannot write frame: VideoWriter is not open.")

        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Frame dimensions {frame.shape[:2]} do not match "
                f"writer dimensions {(self.height, self.width)}."
            )

        self._writer.write(frame)
        self._frames_written += 1

    def __enter__(self) -> AnnotatedVideoWriter:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
