# Demo Scenarios

This document explains the core scenarios you can test with the Road Scene Risk Analyzer and breaks down how the underlying logic interprets the visual data.

> **Disclaimer**: This is a computer-vision prototype. It uses 2D heuristics and does not guarantee production-level ADAS safety.

---

## Scenario 1: Low-Risk Road Scene

**Visual Context**: You are driving on an empty highway, or there are a few vehicles far ahead in the distance.

- **What the detector sees**: YOLO identifies `car` or `truck` bounding boxes, but these boxes are relatively small (occupying a tiny percentage of the total image area) and their bottom coordinates sit high up in the image (closer to the horizon).
- **Risk Calculation**: 
  - Proximity Score is very low (small box + high Y-coordinate).
  - The vehicles are located outside the drawn Danger Zone.
  - No vulnerable users (pedestrians, cyclists) are present.
- **Result**: The scene is classified as **LOW Risk** (Green).

---

## Scenario 2: Medium-Risk Scene

**Visual Context**: You are in moderate traffic. There is a vehicle in the adjacent lane driving relatively close to you, or a large truck ahead of you blocking part of the view.

- **What the detector sees**: YOLO identifies a `car` or `bus` with a moderately large bounding box. The bottom of the box might touch the edge of the Danger Zone or sit slightly above it.
- **Risk Calculation**:
  - Proximity Score is moderate (the box takes up noticeable space).
  - A penalty is added if the object is a "Large Vehicle" (like a bus or truck) positioned near the driving lane, because it obscures visibility.
  - The object is *not* a vulnerable user, and it is *not* directly blocking your path (Danger Zone).
- **Result**: The scene is classified as **MEDIUM Risk** (Yellow).

---

## Scenario 3: High-Risk Scene

**Visual Context**: You are driving in a city. A pedestrian steps into the road right in front of your car, or a cyclist is riding inside your exact driving lane.

- **What the detector sees**: YOLO identifies a `person` or `bicycle`. The bottom-center point of their bounding box falls directly inside the blue trapezoid (the Danger Zone).
- **Risk Calculation**:
  - The system instantly adds a massive **+30 Danger Zone Penalty** because your path is blocked.
  - The system adds another **+25 Vulnerable User Penalty** because pedestrians/cyclists are highly exposed and require immediate braking.
  - Proximity score further amplifies the danger if the box is large (close to the camera).
- **Result**: The scene is classified as **HIGH Risk** (Red) and issues a `CRITICAL` reason string.

---

## System Limitations (What it doesn't know)

While the risk heuristics are effective for a 2D prototype, they have limitations:

1. **No Real Depth Perception**: The system estimates proximity purely based on 2D bounding box area and Y-axis coordinates. It has no LIDAR, radar, or stereo-camera depth sensors. It cannot distinguish between a small car close by and a large truck far away if their 2D footprints look identical.
2. **Fixed Geometric Danger Zone**: The blue Danger Zone is a static trapezoid. It assumes the car is driving straight. It does not dynamically curve with the road or adjust when the driver turns the steering wheel.
3. **No Velocity Tracking**: The system currently evaluates risks frame-by-frame. It does not track an object's trajectory or speed over time (Time-To-Collision).
