"""
Road Scene Risk Analyzer — Global Configuration

Centralizes all project-wide constants, paths, and settings.
Values can be overridden via environment variables or a .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Project Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUTS_DIR = DATA_DIR / "outputs"

# ── YOLO Detection ─────────────────────────────────────────────
YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))

# COCO class names that are relevant for road-scene risk analysis.
# You can uncomment the items below to make the AI detect them.
TARGET_CLASSES: list[str] = [
    # ── High Priority (Moving Road Users) ──
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    
    # ── Static Road Infrastructure ──
    "traffic light",
    "stop sign",
    # "fire hydrant",
    # "parking meter",
    # "bench",
    
    # ── Animals (Potential Road Hazards) ──
    # "dog",
    # "cat",
    # "cow",
    # "horse",
    # "sheep",
    # "bird",
    
    # ── Other Vehicles & Misc ──
    # "train",
    # "umbrella",
    # "backpack",
]

# ── Risk Scoring ───────────────────────────────────────────────
RISK_THRESHOLD_LOW: int = int(os.getenv("RISK_THRESHOLD_LOW", "35"))
RISK_THRESHOLD_HIGH: int = int(os.getenv("RISK_THRESHOLD_HIGH", "70"))

# ── Danger-Zone Polygon (fractions of frame dimensions) ────────
DANGER_ZONE_TOP_LEFT: tuple[float, float] = (0.35, 0.50)
DANGER_ZONE_TOP_RIGHT: tuple[float, float] = (0.65, 0.50)
DANGER_ZONE_BOTTOM_RIGHT: tuple[float, float] = (0.85, 0.95)
DANGER_ZONE_BOTTOM_LEFT: tuple[float, float] = (0.15, 0.95)

# ── Dynamic Lane Detection ─────────────────────────────────────
ENABLE_LANE_DETECTION: bool = os.getenv("ENABLE_LANE_DETECTION", "false").lower() == "true"
LANE_CANNY_LOW: int = int(os.getenv("LANE_CANNY_LOW", "50"))
LANE_CANNY_HIGH: int = int(os.getenv("LANE_CANNY_HIGH", "150"))
LANE_HOUGH_THRESHOLD: int = int(os.getenv("LANE_HOUGH_THRESHOLD", "50"))
LANE_MIN_LINE_LENGTH: int = int(os.getenv("LANE_MIN_LINE_LENGTH", "100"))
LANE_MAX_LINE_GAP: int = int(os.getenv("LANE_MAX_LINE_GAP", "50"))

# ── Accepted Media Formats ─────────────────────────────────────
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS: set[str] = {".mp4", ".avi"}

# ── Video Processing ───────────────────────────────────────────
VIDEO_FRAME_STRIDE: int = int(os.getenv("VIDEO_FRAME_STRIDE", "10"))
