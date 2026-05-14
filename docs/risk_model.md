# Risk Model Specification

> **⚠️ Prototype Notice:** The risk model described here is a **heuristic visual proxy**,
> not a physics-based safety assessment. It must not be used for real-world safety decisions.

---

## 1. Overview

The risk model assigns a **score between 0 and 100** to each detected object in a frame,
then derives a **scene-level classification** (LOW / MEDIUM / HIGH) from the aggregate scores.

The model is:

- **Deterministic** — same input always produces the same output.
- **Explainable** — every score comes with a human-readable `risk_reason` string.
- **Heuristic** — based on 2D bounding-box features, not learned from labeled risk data.

---

## 2. Target Object Classes

Only the following COCO classes are considered for risk scoring:

| Class        | COCO ID | Vulnerable Road User (VRU) |
|--------------|---------|----------------------------|
| `person`     | 0       | ✅ Yes                     |
| `bicycle`    | 1       | ✅ Yes                     |
| `motorcycle` | 3       | ✅ Yes                     |
| `car`        | 2       | ❌ No                      |
| `bus`        | 5       | ❌ No                      |
| `truck`      | 7       | ❌ No                      |

All other COCO classes are filtered out during post-processing.

---

## 3. Object-Level Risk Features

Each detection is characterized by the following features before scoring:

### 3.1 Bounding-Box Area Ratio

```
bbox_area_ratio = (box_width × box_height) / (frame_width × frame_height)
```

**Intuition:** Larger boxes likely correspond to closer objects. This is a **visual size proxy
for proximity**, not a metric distance measurement.

### 3.2 Vertical Position (Normalized)

```
vertical_position = bbox_bottom_y / frame_height
```

**Intuition:** Objects whose bounding-box bottom edge is closer to the bottom of the frame
are typically closer to the camera (in a forward-facing dashcam setup).

### 3.3 Danger-Zone Membership

```
in_danger_zone = point_in_polygon(bottom_center, danger_zone_polygon)
```

The danger zone is a trapezoidal polygon in the lower-center of the frame representing the
road area directly ahead. See [Architecture — §2.4](architecture.md#24-danger-zone-estimation--srcriskdanger_zonepy).

### 3.4 Vulnerable Road User (VRU) Flag

```
is_vru = class_name in {"person", "bicycle", "motorcycle"}
```

VRUs receive a scoring bonus because collisions with unprotected road users are
disproportionately dangerous.

### 3.5 Detector Confidence

```
confidence ∈ [0.0, 1.0]
```

Higher-confidence detections are weighted more heavily to reduce noise from false positives.

---

## 4. Object-Level Risk Score Formula

The score is computed as a **weighted sum** of normalized features, clamped to [0, 100]:

```
raw_score = (w_area     × area_component)
          + (w_position × position_component)
          + (w_zone     × zone_component)
          + (w_vru      × vru_component)
          + (w_conf     × confidence_component)

risk_score = clamp(raw_score, 0, 100)
```

### Default Weights and Components

| Component    | Weight | Value                                               | Range   |
|-------------|--------|-----------------------------------------------------|---------|
| **Area**     | 30     | `min(bbox_area_ratio / 0.15, 1.0) × 30`            | 0–30    |
| **Position** | 25     | `vertical_position × 25`                            | 0–25    |
| **Zone**     | 25     | `25 if in_danger_zone else 0`                       | 0 or 25 |
| **VRU**      | 15     | `15 if is_vru else 0`                               | 0 or 15 |
| **Confidence** | 5   | `confidence × 5`                                    | 0–5     |
| **Total**    | 100    |                                                      | 0–100   |

### Weight Rationale

- **Area (30):** Largest factor because apparent size is the strongest 2D cue for proximity.
- **Position (25):** Bottom-of-frame position strongly correlates with nearness in dashcam footage.
- **Zone (25):** Being in the direct path of travel is a critical risk indicator.
- **VRU (15):** Unprotected users are inherently at higher risk in any collision scenario.
- **Confidence (5):** Minor modulator — we trust the detector but don't want ghost detections
  scoring as high as solid ones.

---

## 5. Risk Reason String

Each detection includes a human-readable `risk_reason` that lists the contributing factors:

```
"Large object (area=0.12), low in frame (pos=0.85), in danger zone, vulnerable road user (person)"
```

This allows a reviewer to understand *why* a particular score was assigned without inspecting
the formula.

---

## 6. Scene-Level Risk Classification

After scoring all detections in a frame, the scene is classified based on the **maximum
object-level risk score** and **aggregate heuristics**:

### Thresholds

| Scene Class | Condition |
|-------------|-----------|
| **HIGH** (≥ 70) | Any object scores ≥ 70, **or** a VRU is inside the danger zone |
| **MEDIUM** (35–69) | Maximum object score is 35–69, **or** ≥ 3 objects in the danger zone |
| **LOW** (< 35) | All object scores < 35 and no VRU in danger zone |

### Scene-Level Score

```
scene_score = max(detection.risk_score for detection in detections) if detections else 0
```

If no objects are detected, the scene defaults to **LOW** with a score of 0.

### Scene Label Mapping

| Score Range | Label    | Color Code | Interpretation                              |
|-------------|----------|------------|---------------------------------------------|
| 0–34        | `LOW`    | 🟢 Green  | No immediate risk indicators                |
| 35–69       | `MEDIUM` | 🟡 Yellow | Caution — objects present near road area     |
| 70–100      | `HIGH`   | 🔴 Red    | Significant risk indicators detected         |

---

## 7. Worked Example

**Frame:** 1280 × 720 dashcam image with two detections.

### Detection A — Pedestrian

| Feature          | Value    | Component Calculation          | Points |
|------------------|----------|--------------------------------|--------|
| `class_name`     | `person` | VRU = yes                      | 15     |
| `confidence`     | 0.91     | 0.91 × 5                      | 4.55   |
| `bbox_area_ratio`| 0.08     | min(0.08/0.15, 1.0) × 30      | 16.0   |
| `vertical_pos`   | 0.82     | 0.82 × 25                     | 20.5   |
| `in_danger_zone` | ✅       | 25                             | 25.0   |
| **Total**        |          |                                | **81.1** |

→ `risk_reason`: "Large object (area=0.08), low in frame (pos=0.82), in danger zone, vulnerable road user (person)"

### Detection B — Car

| Feature          | Value | Component Calculation       | Points |
|------------------|-------|-----------------------------|--------|
| `class_name`     | `car` | VRU = no                    | 0      |
| `confidence`     | 0.87  | 0.87 × 5                   | 4.35   |
| `bbox_area_ratio`| 0.03  | min(0.03/0.15, 1.0) × 30   | 6.0    |
| `vertical_pos`   | 0.55  | 0.55 × 25                  | 13.75  |
| `in_danger_zone` | ❌    | 0                           | 0      |
| **Total**        |       |                             | **24.1** |

→ `risk_reason`: "Small object (area=0.03), mid-frame (pos=0.55)"

### Scene Classification

- `scene_score = max(81.1, 24.1) = 81.1`
- `scene_class = HIGH` (≥ 70, and a VRU is in the danger zone)

---

## 8. Danger-Zone Polygon

The default danger zone is a **trapezoid** in the lower-center of the frame:

```
        top_left ──────── top_right
          /                    \
         /                      \
bottom_left ──────── bottom_right
```

Default parameters (as fractions of frame dimensions):

| Vertex        | x (fraction of width) | y (fraction of height) |
|---------------|-----------------------|------------------------|
| `top_left`    | 0.35                  | 0.50                   |
| `top_right`   | 0.65                  | 0.50                   |
| `bottom_right`| 0.85                  | 0.95                   |
| `bottom_left` | 0.15                  | 0.95                   |

These can be adjusted in `src/config.py` to fit different camera angles.

---

## 9. Limitations

> [!IMPORTANT]
> This section must be read before interpreting any output from the system.

1. **No real depth estimation.** Monocular 2D images cannot provide reliable metric distance
   without camera calibration or a depth-estimation model. The `bbox_area_ratio` and
   `vertical_position` features are **visual proxies**, not physical measurements.

2. **Static danger zone.** The polygon does not adapt to road curvature, lane markings,
   or varying camera mounting positions.

3. **No temporal context.** Each frame is scored independently. There is no object tracking,
   velocity estimation, or trajectory prediction. A stationary car and a car approaching at
   high speed receive the same score if their bounding boxes are identical.

4. **COCO class limitations.** The detector may miss road-specific objects like traffic cones,
   barriers, or construction signs that are not in the COCO class set.

5. **Lighting and weather.** Model performance degrades in poor visibility conditions
   (night, rain, fog, glare) that are underrepresented in COCO training data.

6. **No calibration.** The system does not use camera intrinsics or extrinsics.
   Results will vary across different dashcam models and mounting positions.

7. **Not a safety system.** This prototype has not been validated against ISO 26262, SOTIF,
   or any automotive safety standard. It is intended for educational and portfolio purposes only.

---

*Last updated: 2026-05-14*
