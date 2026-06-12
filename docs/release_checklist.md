# Release Checklist

> Final pre-release verification for the Road Scene Risk Analyzer portfolio project.

---

## ✅ Installation & Setup

- [x] Repository clones successfully from a clean `git clone`
- [x] `python -m venv .venv && pip install -r requirements.txt` completes without errors
- [x] `.env.example` is present and documented
- [x] `.gitignore` excludes model weights, virtual environments, outputs, and caches
- [x] No generated outputs, model weights, large datasets, or virtual environments are committed

---

## ✅ Application Startup

- [x] `streamlit run app/streamlit_app.py` launches without errors
- [x] Dashboard loads at `http://localhost:8501`
- [x] YOLO model weights auto-download on first run
- [x] Sidebar settings (confidence, danger zone, stride) are functional

---

## ✅ Image Demo

- [x] Upload a `.jpg` or `.png` image → analysis completes
- [x] Annotated image displays with bounding boxes, risk labels, and danger zone overlay
- [x] Scene risk panel shows risk level (LOW / MEDIUM / HIGH) with reason
- [x] Detection table lists all detected objects with class, confidence, risk score
- [x] Download buttons work for annotated image (PNG), JSON report, CSV table

---

## ✅ Video Demo

- [x] Upload a `.mp4` or `.avi` video → frame-by-frame analysis completes
- [x] Summary metrics display: peak risk, max score, high-risk frames, avg objects/frame
- [x] Riskiest frame is extracted and displayed with annotations
- [x] Download buttons work for JSON report and CSV table
- [ ] ⚠️ **Limitation documented:** Full annotated video export is not yet supported

---

## ✅ Screenshots & Demo Assets

- [x] At least one example result is visible in README (ASCII art + JSON sample)
- [x] `docs/examples/sample_report.json` — example JSON output
- [x] `docs/examples/sample_detections.csv` — example CSV output
- [x] `docs/assets/` directory exists for future screenshot storage

---

## ✅ Output Exports

- [x] JSON export is valid JSON with all schema fields
- [x] CSV export has correct header and row format
- [x] Image export produces valid PNG bytes
- [x] Filenames include risk level and timestamp

---

## ✅ Tests

- [x] `python -m pytest tests/ -v` — all 106 tests pass
- [x] **Danger zone tests:** polygon creation, point-in-polygon, edge cases
- [x] **Scoring tests:** pedestrian in zone = HIGH, distant car = LOW, VRU bonus, score clamping
- [x] **Scene classifier tests:** threshold boundaries (LOW/MEDIUM/HIGH), reason strings
- [x] **Exporter tests:** JSON/CSV shape, image encoding, empty detections
- [x] **Annotator tests:** immutability, missing scene_risk handling
- [x] **Smoke tests:** imports, Detection schema, detector instantiation
- [x] **Integration tests:** image pipeline, video pipeline

---

## ✅ README & Documentation

- [x] README contains project title and one-sentence pitch
- [x] README includes example output (ASCII diagram + JSON)
- [x] README has feature list table
- [x] README has Mermaid architecture diagram
- [x] README has copy-pasteable installation steps
- [x] README has usage steps (Streamlit + Python API)
- [x] README explains the risk model with score component table
- [x] README lists 7 explicit limitations
- [x] README has project structure tree
- [x] README has configuration reference
- [x] README has tech stack table
- [x] README lists 8 possible improvements
- [x] README has CV bullet
- [x] README has ethical/legal note with CAUTION alert
- [x] `docs/architecture.md` — pipeline architecture
- [x] `docs/risk_model.md` — scoring formula and thresholds
- [x] `docs/demo_scenarios.md` — expected behavior for 3 scenarios
- [x] `docs/data_strategy.md` — data sources and exclusions
- [x] `docs/technical_report.md` — design decisions and rationale

---

## ✅ Known Limitations (Honest & Specific)

The following limitations are explicitly documented in the README and technical report:

1. **No real depth estimation** — 2D bounding-box heuristics only
2. **Static danger zone** — does not adapt to road curvature or steering
3. **No temporal tracking** — no object tracking, velocity, or TTC across frames
4. **Single camera assumption** — forward-facing dashcam only
5. **COCO class limitations** — may miss traffic cones, barriers, road markings
6. **Weather/lighting degradation** — accuracy drops in rain, fog, night, glare
7. **No camera calibration** — no intrinsic/extrinsic parameters used
8. **Not a safety system** — not validated against ISO 26262, SOTIF, or any standard

---

## ✅ Future Work

Documented improvements for potential follow-up:

| Item | Status |
|------|--------|
| Monocular depth estimation (MiDaS/ZoeDepth) | Documented |
| Object tracking (DeepSORT/ByteTrack) | Documented |
| Dynamic lane detection | Documented |
| Fine-tuned model on road datasets | Documented |
| Night/weather preprocessing | Documented |
| Batch processing CLI | Documented |
| Docker deployment | Documented |
| Cloud deployment (Streamlit Cloud/HuggingFace) | Documented |

---

## ✅ Repository Hygiene

- [x] No `.pyc`, `__pycache__`, or `.pytest_cache` committed
- [x] No `.venv` or `venv` committed
- [x] No `*.pt` model weight files committed
- [x] No `data/outputs/` committed
- [x] No large image/video files in root committed
- [x] `.gitignore` is comprehensive
- [x] No sensitive data (API keys, personal info) in any committed file

---

## 🎤 Interview Ready

> **60-second pitch:**
>
> "I built a computer-vision prototype called Road Scene Risk Analyzer. It uses
> YOLOv8 to detect vehicles and pedestrians in dashcam images, maps a danger zone
> polygon representing the driving path, and computes a heuristic risk score for
> each object based on proximity, position, and class type. The system classifies
> scenes as LOW, MEDIUM, or HIGH risk and provides explanations for each score.
> It includes a Streamlit dashboard for interactive analysis and over 100 unit tests.
> The key design decision was using a deterministic heuristic scoring model instead
> of a learned model — this makes every score fully explainable and testable.
> I was careful to document all limitations explicitly — it's a 2D prototype
> without depth estimation or object tracking, not a production safety system."

---

*Checklist completed: 2026-06-12*
