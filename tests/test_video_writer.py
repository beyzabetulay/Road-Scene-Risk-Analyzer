import numpy as np
import pytest
import os
import cv2
from src.io.video_writer import AnnotatedVideoWriter

def test_video_writer_creates_file(tmp_path):
    output_path = tmp_path / "test_out.mp4"
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    with AnnotatedVideoWriter(output_path, 30.0, 100, 100) as writer:
        writer.write_frame(frame)
        writer.write_frame(frame)
        
    assert output_path.exists()
    assert output_path.stat().st_size > 0

def test_video_writer_wrong_dimensions(tmp_path):
    output_path = tmp_path / "test_out.mp4"
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    
    with pytest.raises(ValueError):
        with AnnotatedVideoWriter(output_path, 30.0, 100, 100) as writer:
            writer.write_frame(frame)

def test_video_writer_not_open_raises_error(tmp_path):
    output_path = tmp_path / "test_out.mp4"
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    writer = AnnotatedVideoWriter(output_path, 30.0, 100, 100)
    with pytest.raises(RuntimeError):
        writer.write_frame(frame)
