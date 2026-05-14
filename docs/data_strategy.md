# Data Strategy

> **⚠️ Prototype Notice:** Road Scene Risk Analyzer is a computer-vision research prototype.
> It is **not** a production-grade ADAS (Advanced Driver Assistance System) module and must
> not be used for real-world safety-critical decisions.

---

## 1. Accepted Input Formats

The pipeline accepts the following file types through both the Streamlit UI and the CLI:

| Type   | Extensions         | Notes                                         |
|--------|--------------------|-----------------------------------------------|
| Image  | `.jpg`, `.jpeg`, `.png` | Single-frame analysis                     |
| Video  | `.mp4`, `.avi`     | Frame-by-frame processing at configurable FPS |

All other formats are rejected at load time with an informative error message.
See `src/io/media_loader.py` for the loading logic.

---

## 2. Sample Data Sources

The MVP intentionally avoids shipping large or license-restricted datasets.
Sample images and videos placed under `data/samples/` should come from one of these sources:

### Recommended

| Source | Why |
|--------|-----|
| **Your own dashcam footage** | No licensing issues; most realistic test data. |
| **Self-created synthetic images** | Drawings, renders, or screenshots from open simulators (e.g., CARLA). |
| **Public-domain / CC-0 images** | Sites like [Unsplash](https://unsplash.com), [Pexels](https://pexels.com), or [Pixabay](https://pixabay.com) offer traffic photos under permissive licenses. Always verify the specific license. |

### Acceptable with Caution

| Source | Condition |
|--------|-----------|
| **KITTI** ([www.cvlibs.net/datasets/kitti](http://www.cvlibs.net/datasets/kitti/)) | Only if you comply with the [Creative Commons BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) license. Do **not** redistribute in this repo. |
| **Cityscapes** ([www.cityscapes-dataset.com](https://www.cityscapes-dataset.com/)) | Requires registration and agreement to their terms. Do **not** redistribute. |

### Not Acceptable

- Images scraped from the internet without explicit license.
- Footage containing identifiable faces or license plates (privacy risk).
- Any file that would violate copyright or data-protection regulations.

### How to Add Your Own Samples

```bash
# Copy images into the samples directory
cp ~/dashcam/frame_001.jpg data/samples/

# Copy a short video clip
cp ~/dashcam/clip.mp4 data/samples/

# Then run the Streamlit app or point the pipeline at the file
streamlit run app/streamlit_app.py
```

The Streamlit dashboard also supports **drag-and-drop upload** — no file copying required.

---

## 3. Why COCO-Pretrained Detection (No Custom Training)

The MVP uses a **YOLOv8 model pretrained on the COCO dataset** instead of training a custom model. Reasons:

1. **Immediate results.** COCO covers 80 common object classes including `car`, `truck`, `bus`, `person`, `bicycle`, and `motorcycle` — the most relevant categories for road-scene analysis.
2. **No GPU training required.** Users can run inference on CPU or a modest GPU without a training pipeline.
3. **Reproducibility.** Everyone gets the same baseline model weights from Ultralytics, eliminating "works on my machine" issues.
4. **Scope control.** Custom training introduces hyperparameter tuning, data augmentation pipelines, and evaluation infrastructure that are out of scope for this prototype.

If higher accuracy on domain-specific classes (e.g., traffic signs, road barriers) is needed in the future, fine-tuning on a curated road-scene dataset is the recommended next step.

---

## 4. What Is **Not** Included

To set clear expectations, this project explicitly does **not** provide:

| Exclusion | Reason |
|-----------|--------|
| **Dataset redistribution** | No KITTI, Cityscapes, BDD100K, or other dataset files are shipped in this repo. Users must source their own samples. |
| **Real safety certification** | This is a portfolio/research prototype. It has not been validated against ISO 26262, SOTIF, or any automotive safety standard. |
| **Personal-data collection** | The app processes files locally. It does not upload, store, or transmit user data to any server. |
| **Production-grade latency guarantees** | Frame processing speed depends on hardware. No real-time guarantees are made. |
| **Model training scripts** | Only inference is supported. Training pipelines are out of scope for the MVP. |

---

## 5. Git & Repository Hygiene

The `.gitignore` is configured to prevent accidental commits of:

- Model weight files (`*.pt`, `*.onnx`, `*.engine`)
- Large video files (except whitelisted samples under `data/samples/`)
- Generated outputs (`data/outputs/`)
- Virtual environments and caches

If you add sample files, keep them **small** (a few MB each). For video, prefer short clips (5–15 seconds).

---

## 6. Dataset Limitations

- **Class coverage:** COCO-pretrained models may miss domain-specific objects (e.g., traffic cones, road barriers, construction signs).
- **Weather / lighting:** Model performance may degrade in rain, fog, night, or glare conditions not well-represented in COCO.
- **Geographic bias:** COCO data is skewed toward North American and European road scenes. Performance on other regions is untested.
- **Scale:** The MVP is designed for single-image or short-video analysis, not large-scale batch processing of thousands of frames.

---

*Last updated: 2026-05-14*
