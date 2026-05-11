# 🚗 Road Scene Risk Analyzer

A computer-vision prototype that detects objects in road/traffic scenes, estimates danger zones, and assigns risk scores — all visualized through an interactive Streamlit dashboard.

> **⚠️ Disclaimer:** This is a CV research prototype, **not** a production-level ADAS safety system.

---

## 🎯 Overview

| Feature | Description |
|---|---|
| **Object Detection** | YOLOv8-based detection of vehicles, pedestrians, cyclists, etc. |
| **Risk Scoring** | Proximity and trajectory-based danger scoring per detected object |
| **Scene Classification** | Categorize scenes (highway, intersection, school zone, …) |
| **Danger Zone Mapping** | Polygon-based overlay of high-risk regions |
| **Streamlit Dashboard** | Upload images/videos and view annotated results interactively |
| **Export** | Save annotated frames, CSV reports, and JSON summaries |

---

## 📂 Project Structure

```
road-scene-risk-analyzer/
├── app/
│   └── streamlit_app.py          # Streamlit entry point
├── src/
│   ├── config.py                 # Global configuration & constants
│   ├── detection/
│   │   ├── detector.py           # YOLO-based object detector
│   │   └── schemas.py            # Detection data schemas
│   ├── risk/
│   │   ├── danger_zone.py        # Danger zone polygon logic
│   │   ├── scoring.py            # Risk scoring engine
│   │   └── scene_classifier.py   # Scene type classification
│   ├── visualization/
│   │   └── annotator.py          # Frame annotation & overlays
│   ├── io/
│   │   ├── media_loader.py       # Image/video loading utilities
│   │   └── exporters.py          # CSV, JSON, image export
│   └── utils/
│       └── logging.py            # Logging configuration
├── tests/                        # pytest test suite
├── data/
│   ├── samples/                  # Sample input images/videos
│   └── outputs/                  # Generated outputs (git-ignored)
├── docs/                         # Documentation
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & set up virtual environment

```bash
git clone https://github.com/beyzabetulay/Road-Scene-Risk-Analyzer.git
cd Road-Scene-Risk-Analyzer

python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### 2. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

### 3. Run tests

```bash
python -m pytest
```

---

## ⚙️ Configuration

Copy the example environment file and fill in your settings:

```bash
cp .env.example .env
```

See `src/config.py` for available configuration options.

---

## 🛠️ Tech Stack

- **Detection:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Image Processing:** OpenCV, Pillow
- **UI:** Streamlit
- **Data:** NumPy, Pandas
- **Testing:** pytest

---

## 📄 License

This project is for educational / portfolio purposes.
