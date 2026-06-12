# 🚗 Road Scene Risk Analyzer

> **Real-time road scene analysis using computer vision — detects vehicles and pedestrians, maps danger zones, and assigns heuristic risk scores through an interactive Streamlit dashboard.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen?logo=yolo)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

> **⚠️ Disclaimer:** This is a computer-vision research prototype, **not** a production-level ADAS safety system. See [Limitations](#-limitations) and [Ethical & Legal Note](#%EF%B8%8F-ethical--legal-note).

---

## 📸 Example Output

Below is an example of the analyzer processing a road scene:

```
┌──────────────────────────────────────────────────────────────┐
│  SCENE RISK: HIGH                                            │
│  CRITICAL: Person inside estimated driving lane!             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│              ┌─────────┐                                     │
│              │ PERSON   │ Risk: 82                            │
│              │ (0.91)   │                                     │
│              └─────────┘                                     │
│         ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲                              │
│        ╱    DANGER ZONE        ╲                             │
│       ╱                         ╲                            │
│      ╱___________________________╲                           │
│                                                              │
│  ┌──────┐           ┌──────┐                                 │
│  │ CAR  │           │ CAR  │  Risk: 24                       │
│  │(0.87)│           │(0.65)│                                 │
│  └──────┘           └──────┘                                 │
└──────────────────────────────────────────────────────────────┘
```

**Sample JSON output:**
```json
{
  "scene_risk": {
    "risk_level": "HIGH",
    "max_risk_score": 82.0,
    "reason": "CRITICAL: Person inside estimated driving lane!"
  },
  "detection_count": 3,
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.91,
      "risk_score": 82.0,
      "in_danger_zone": true,
      "risk_reason": "Vulnerable user in driving lane"
    },
    {
      "class_name": "car",
      "confidence": 0.87,
      "risk_score": 24.1,
      "in_danger_zone": false,
      "risk_reason": "Object detected (outside lane)"
    }
  ]
}
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Object Detection** | YOLOv8 (nano) detecting vehicles, pedestrians, cyclists, traffic lights, and stop signs |
| 📊 **Risk Scoring** | Heuristic 0–100 score per object based on proximity, position, danger zone, and class type |
| 🏷️ **Scene Classification** | Automatic LOW / MEDIUM / HIGH scene-level risk classification |
| 🔷 **Danger Zone Mapping** | Configurable trapezoidal polygon overlay representing the driving path |
| 🖥️ **Streamlit Dashboard** | Interactive web UI for uploading images/videos and viewing annotated results |
| 📥 **Export** | Download annotated images, JSON reports, and CSV detection tables |
| 🎥 **Video Support** | Frame-by-frame video analysis with configurable stride |
| ⚙️ **Configurable** | Adjust confidence thresholds, danger zone shape, and model variant via UI or `.env` |

---

## 🏗️ Architecture

```mermaid
flowchart LR
    A["📷 Image / Video"] --> B["Media Loader"]
    B --> C["YOLOv8 Detector"]
    C --> D["Detection Post-Processing"]
    D --> E["Danger Zone Check"]
    E --> F["Risk Scoring Engine"]
    F --> G["Scene Classifier"]
    G --> H["Annotator"]
    H --> I["Exporter (JSON/CSV/Image)"]
    I --> J["Streamlit Dashboard"]

    style A fill:#e1f5fe
    style J fill:#e8f5e9
    style F fill:#fff3e0
```

### Pipeline Overview

1. **Media Loader** — Accepts `.jpg`, `.jpeg`, `.png` images or `.mp4`, `.avi` videos
2. **YOLOv8 Detector** — Runs COCO-pretrained object detection (nano model for fast CPU inference)
3. **Post-Processing** — Filters detections by target classes and confidence threshold
4. **Danger Zone** — Tests each object's bottom-center against a trapezoidal polygon
5. **Risk Scoring** — Computes a 0–100 score from area ratio, Y-position, zone membership, VRU status, and confidence
6. **Scene Classifier** — Maps the maximum object score to LOW (<35) / MEDIUM (35–69) / HIGH (≥70)
7. **Annotator** — Draws bounding boxes, risk labels, and danger zone overlay on frames
8. **Exporter** — Generates downloadable JSON reports, CSV tables, and annotated images

For detailed architecture documentation, see [docs/architecture.md](docs/architecture.md).

---

## 🚀 Installation

### Prerequisites

- **Python 3.10+**
- **pip** package manager
- ~200 MB disk space (for YOLOv8 nano weights, auto-downloaded on first run)

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/beyzabetulay/Road-Scene-Risk-Analyzer.git
cd Road-Scene-Risk-Analyzer

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. (Optional) Copy environment config
cp .env.example .env
```

### Verify Installation

```bash
# Run the test suite (all tests should pass)
python -m pytest tests/ -v
```

---

## 🎮 Usage

### Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`. From there you can:

1. **Upload** a dashcam image (`.jpg`, `.png`) or short video (`.mp4`, `.avi`)
2. **Adjust settings** in the sidebar — confidence threshold, danger zone shape, frame stride
3. **Click "🚀 Analyze"** to run the pipeline
4. **View results** — annotated image, risk metrics, detection table
5. **Download** annotated image, JSON report, or CSV detection table

### Quick Test with Python

```python
from src.pipeline import analyze_image

result = analyze_image("sample_road.jpg")

print(f"Scene Risk: {result.scene_risk.risk_level}")
print(f"Max Score:  {result.scene_risk.max_risk_score}")
print(f"Detections: {result.detection_count}")

for det in result.detections:
    print(f"  {det.class_name}: risk={det.risk_score}, in_zone={det.in_danger_zone}")
```

---

## 📐 Risk Model Explanation

The risk scoring engine uses a **deterministic, heuristic formula** — no machine-learned risk model is involved.

### Per-Object Score (0–100)

| Component | Max Points | Description |
|-----------|-----------|-------------|
| **Area Ratio** | 20 | Larger bounding box → likely closer to camera |
| **Y-Position** | 15 | Lower in frame → closer (dashcam perspective) |
| **Danger Zone** | 30 | Object's bottom-center inside the driving path polygon |
| **Vulnerable User** | 25 | `person`, `bicycle`, or `motorcycle` bonus |
| **Large Vehicle** | 10 | `bus` or `truck` near the bottom of the frame |
| **Confidence** | 5 | Higher detector confidence → slightly higher certainty |

### Scene Classification

| Level | Score Range | Meaning |
|-------|-------------|---------|
| 🟢 **LOW** | 0–34 | No close objects in danger zone |
| 🟡 **MEDIUM** | 35–69 | Objects present but not critically blocking path |
| 🔴 **HIGH** | ≥ 70 | Vulnerable user or vehicle directly in driving lane |

For the full formula, worked examples, and weight rationale, see [docs/risk_model.md](docs/risk_model.md).

---

## ⚠️ Limitations

> These limitations are inherent to the 2D-heuristic approach and must be understood before interpreting any output.

1. **No real depth estimation.** All "proximity" signals are derived from 2D bounding-box size and vertical position. Without LiDAR, radar, or stereo cameras, metric distance cannot be reliably measured.

2. **Static danger zone.** The trapezoidal polygon does not adapt to road curvature, lane markings, or steering angle. It assumes a straight driving path.

3. **No temporal tracking.** Each frame is scored independently — no object tracking, velocity estimation, or Time-to-Collision (TTC) calculation across frames.

4. **Single camera assumption.** The pipeline assumes a single forward-facing dashcam. Other viewpoints will produce unreliable results.

5. **COCO class limitations.** The detector may miss domain-specific objects like traffic cones, construction barriers, or road markings not in the COCO class set.

6. **Weather and lighting.** Model accuracy degrades in poor visibility (night, rain, fog, glare) and with occluded or unusually-posed objects.

7. **No camera calibration.** No intrinsic or extrinsic camera parameters are used. Results vary across different dashcam models and mounting positions.

---

## 📂 Project Structure

```
Road-Scene-Risk-Analyzer/
├── app/
│   └── streamlit_app.py            # Streamlit web dashboard entry point
├── src/
│   ├── config.py                   # Global configuration & constants
│   ├── pipeline.py                 # Analysis orchestration (image & video)
│   ├── detection/
│   │   ├── detector.py             # YOLO-based object detector wrapper
│   │   └── schemas.py              # Detection dataclass schema
│   ├── risk/
│   │   ├── danger_zone.py          # Polygon-based danger zone logic
│   │   ├── scoring.py              # Per-object risk scoring engine
│   │   └── scene_classifier.py     # Scene-level risk classification
│   ├── visualization/
│   │   └── annotator.py            # Frame annotation & overlays
│   ├── io/
│   │   ├── media_loader.py         # Image/video loading utilities
│   │   └── exporters.py            # JSON, CSV, image export
│   └── utils/
│       └── logging.py              # Logging configuration
├── tests/                          # pytest test suite (100+ tests)
│   ├── test_danger_zone.py         # Danger zone polygon & point tests
│   ├── test_scoring.py             # Risk scoring engine tests
│   ├── test_scene_classifier.py    # Scene classification threshold tests
│   ├── test_exporters.py           # JSON/CSV/image export tests
│   ├── test_annotator.py           # Visualization tests
│   ├── test_smoke.py               # Import & detector smoke tests
│   ├── test_image_pipeline.py      # Image pipeline integration tests
│   └── test_video_pipeline.py      # Video pipeline integration tests
├── docs/
│   ├── architecture.md             # Pipeline architecture documentation
│   ├── risk_model.md               # Risk model specification
│   ├── demo_scenarios.md           # Expected behavior for demo scenarios
│   └── data_strategy.md            # Data sources & strategy
├── data/
│   ├── samples/                    # Sample input images/videos
│   └── outputs/                    # Generated outputs (git-ignored)
├── requirements.txt                # Python dependencies
├── .env.example                    # Example environment configuration
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## ⚙️ Configuration

All tunable parameters are centralized in `src/config.py` and can be overridden via environment variables or `.env`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `YOLO_MODEL` | `yolov8n.pt` | Model variant (nano/small/medium/large) |
| `CONFIDENCE_THRESHOLD` | `0.25` | Minimum detector confidence |
| `TARGET_CLASSES` | 8 COCO classes | Classes to detect (person, car, bus, etc.) |
| `RISK_THRESHOLD_LOW` | `35` | Below this → LOW scene risk |
| `RISK_THRESHOLD_HIGH` | `70` | At or above this → HIGH scene risk |
| `DANGER_ZONE_*` | *(see config)* | Trapezoidal polygon shape parameters |
| `VIDEO_FRAME_STRIDE` | `10` | Process every N-th frame in video mode |

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Object Detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | COCO-pretrained real-time detection |
| Image Processing | OpenCV, Pillow | Frame loading, encoding, polygon tests |
| Web Dashboard | [Streamlit](https://streamlit.io/) | Interactive upload, visualization, download |
| Data Handling | NumPy, Pandas | Array operations, detection tables |
| Testing | pytest | 100+ unit and integration tests |
| Configuration | python-dotenv | Environment variable management |

---

## 🔮 Possible Improvements

| Improvement | Description |
|-------------|-------------|
| 🎯 **Fine-tuned model** | Train on a road-specific dataset (BDD100K, Waymo) for better class coverage |
| 📏 **Monocular depth estimation** | Add MiDaS or ZoeDepth for metric distance estimation |
| 🔄 **Object tracking** | Implement DeepSORT or ByteTrack for cross-frame identity and TTC |
| 🛣️ **Lane detection** | Replace the static danger zone with dynamic lane-aware polygons |
| 🌙 **Night/weather adaptation** | Add preprocessing for low-light and adverse weather conditions |
| 📊 **Batch processing** | Add CLI mode for processing directories of images/videos |
| 🐳 **Docker deployment** | Containerize the application for reproducible deployment |
| ☁️ **Cloud deployment** | Deploy to Streamlit Cloud or Hugging Face Spaces for public demo |

---

## 📝 CV Bullet

> **Road Scene Risk Analyzer** — Built a computer-vision prototype using YOLOv8 and OpenCV that detects road objects, maps danger zones with polygon-based heuristics, and classifies scene risk (LOW/MEDIUM/HIGH) through a Streamlit dashboard. Includes 100+ pytest unit tests and full architecture documentation.

---

## ⚖️ Ethical & Legal Note

> [!CAUTION]
> **This project is a computer-vision research prototype** and is **NOT** a production-grade ADAS (Advanced Driver Assistance System) module.

- **Not validated** against any automotive safety standard (ISO 26262, SOTIF, UN R157, etc.)
- **Not intended** for real-world safety-critical decisions, autonomous driving, or driver alerting
- **No dataset redistribution** — users must source their own sample images/videos
- **No personal data collection** — all processing is local; no data is uploaded or transmitted
- **No real-time guarantees** — processing speed depends on hardware and is not bounded
- **COCO-pretrained model** — the detector is general-purpose, not specialized for road safety

Use this project only for **educational, research, and portfolio** purposes.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | Pipeline modules, data flow, configuration, technology choices |
| [Risk Model](docs/risk_model.md) | Heuristic scoring formula, thresholds, worked example, limitations |
| [Demo Scenarios](docs/demo_scenarios.md) | Expected behavior for low, medium, and high-risk road scenes |
| [Data Strategy](docs/data_strategy.md) | Accepted formats, sample sources, why COCO, exclusions |
| [Technical Report](docs/technical_report.md) | Design decisions, trade-offs, and architectural rationale |

---

## 📄 License

This project is for **educational and portfolio purposes** only.

---

*Built with ❤️ by [Beyza Betül AY](https://github.com/beyzabetulay)*
