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
TARGET_CLASSES: list[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
]

# Optional classes — included when INCLUDE_OPTIONAL_CLASSES is set.
OPTIONAL_CLASSES: list[str] = [
    "traffic light",
    "stop sign",
]

INCLUDE_OPTIONAL_CLASSES: bool = (
    os.getenv("INCLUDE_OPTIONAL_CLASSES", "false").lower() == "true"
)

if INCLUDE_OPTIONAL_CLASSES:
    TARGET_CLASSES = TARGET_CLASSES + OPTIONAL_CLASSES

# ── Risk Scoring ───────────────────────────────────────────────
RISK_THRESHOLD_LOW: int = int(os.getenv("RISK_THRESHOLD_LOW", "35"))
RISK_THRESHOLD_HIGH: int = int(os.getenv("RISK_THRESHOLD_HIGH", "70"))

# ── Danger-Zone Polygon (fractions of frame dimensions) ────────
DANGER_ZONE_TOP_LEFT: tuple[float, float] = (0.35, 0.50)
DANGER_ZONE_TOP_RIGHT: tuple[float, float] = (0.65, 0.50)
DANGER_ZONE_BOTTOM_RIGHT: tuple[float, float] = (0.85, 0.95)
DANGER_ZONE_BOTTOM_LEFT: tuple[float, float] = (0.15, 0.95)

# ── Accepted Media Formats ─────────────────────────────────────
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS: set[str] = {".mp4", ".avi"}
