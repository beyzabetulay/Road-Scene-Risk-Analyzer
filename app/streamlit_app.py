"""
Road Scene Risk Analyzer — Streamlit UI
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path so 'src' can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    CONFIDENCE_THRESHOLD,
    DANGER_ZONE_BOTTOM_LEFT,
    DANGER_ZONE_BOTTOM_RIGHT,
    DANGER_ZONE_TOP_LEFT,
    DANGER_ZONE_TOP_RIGHT,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    VIDEO_FRAME_STRIDE,
    YOLO_MODEL,
)
from src.pipeline import analyze_image, analyze_video
from src.risk.danger_zone import DangerZoneParams
from src.visualization.annotator import annotate_image
from src.io.exporters import (
    export_image_to_bytes,
    export_report_to_json,
    export_table_to_csv,
    generate_export_filename,
)


st.set_page_config(
    page_title="Road Scene Risk Analyzer",
    page_icon="🚘",
    layout="wide",
)

# ── Helpers ─────────────────────────────────────────────────────────


def save_uploaded_file(uploaded_file) -> str:
    """Save Streamlit UploadedFile to a temporary file and return the path."""
    suffix = Path(uploaded_file.name).suffix
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def get_video_frame(video_path: str, frame_index: int) -> np.ndarray | None:
    """Extract a specific frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    return None


# ── UI ─────────────────────────────────────────────────────────────

st.title("🚘 Road Scene Risk Analyzer")
st.markdown(
    """
    Welcome to the Road Scene Risk Analyzer prototype. Upload a dashcam image 
    or short video to analyze the road scene. The system will detect vehicles 
    and pedestrians, map a heuristic "danger zone" for the driving lane, and 
    assign risk scores based on proximity and object type.
    """
)

with st.expander("📖 How does this work? (Demo Scenarios & Logic)"):
    st.markdown(
        """
        **The system evaluates risk based on 2D heuristics:**
        - 🟢 **LOW Risk**: Vehicles are far away, outside the blue Danger Zone, and occupy a small portion of the screen.
        - 🟡 **MEDIUM Risk**: A vehicle is close to the driving lane, or a large vehicle (bus/truck) is blocking visibility, but your immediate path is clear.
        - 🔴 **HIGH Risk**: A vulnerable user (pedestrian/cyclist) or vehicle is directly inside your estimated path (the blue Danger Zone).
        
        **Limitations:**
        This is a computer-vision prototype. It lacks real depth sensors (LIDAR/Radar) and relies entirely on 2D bounding box sizes and Y-coordinates to guess proximity. It does not track velocity (Time-To-Collision) and assumes a straight driving path.
        """
    )

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    model_name = st.text_input("YOLO Model", value=YOLO_MODEL)
    
    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
    )
    
    draw_zone = st.checkbox("Draw Danger Zone Overlay", value=True)
    
    st.subheader("Danger Zone Params")
    st.caption("Adjust the driving lane trapezoid")
    
    col1, col2 = st.columns(2)
    with col1:
        top_w = st.slider("Top Width", 0.1, 1.0, 0.3, step=0.05)
    with col2:
        top_y = st.slider("Top Y", 0.1, 1.0, 0.5, step=0.05)
        
    col3, col4 = st.columns(2)
    with col3:
        bot_w = st.slider("Bottom Width", 0.1, 1.0, 0.7, step=0.05)
    with col4:
        bot_y = st.slider("Bottom Y", 0.5, 1.0, 0.95, step=0.05)
        
    st.subheader("Video Params")
    stride = st.slider(
        "Frame Stride",
        min_value=1,
        max_value=60,
        value=VIDEO_FRAME_STRIDE,
        help="Process every N-th frame to speed up video analysis.",
    )
    
    st.subheader("Advanced Features")
    from src.config import ENABLE_LANE_DETECTION, ENABLE_DEPTH_ESTIMATION
    use_lane_detection = st.toggle(
        "Dynamic Lane Detection",
        value=ENABLE_LANE_DETECTION,
        help="Use OpenCV to dynamically detect lane lines and adjust the Danger Zone.",
    )
    use_depth = st.toggle(
        "Depth Estimation (MiDaS)",
        value=ENABLE_DEPTH_ESTIMATION,
        help="Use MiDaS model to estimate object distance and improve risk scoring.",
    )

# Compute custom danger zone coordinates
center_x = 0.5
dz_params = DangerZoneParams(
    top_left=(center_x - top_w / 2, top_y),
    top_right=(center_x + top_w / 2, top_y),
    bottom_right=(center_x + bot_w / 2, bot_y),
    bottom_left=(center_x - bot_w / 2, bot_y),
)


# Main layout
uploaded_file = st.file_uploader(
    "Choose an image or video...",
    type=[ext.replace(".", "") for ext in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS],
)

if uploaded_file is not None:
    file_ext = Path(uploaded_file.name).suffix.lower()
    is_video = file_ext in VIDEO_EXTENSIONS
    
    if st.button("🚀 Analyze", type="primary"):
        with st.spinner("Analyzing... Please wait."):
            tmp_path = save_uploaded_file(uploaded_file)
            
            try:
                if not is_video:
                    # IMAGE ANALYSIS
                    result = analyze_image(
                        tmp_path,
                        model_name=model_name,
                        confidence_threshold=conf_thresh,
                        source_name=uploaded_file.name,
                        danger_zone_params=dz_params,
                        use_depth=use_depth,
                    )
                    
                    # Read original image to pass to annotator
                    image_bgr = cv2.imread(tmp_path)
                    annotated_bgr = annotate_image(
                        image_bgr, result, draw_danger_zone=draw_zone
                    )
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    
                    # Show Image
                    st.image(annotated_rgb, caption="Analysis Result", use_container_width=True)
                    
                    # Show Summary
                    st.subheader("📊 Scene Summary")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    metrics_col1.metric("Risk Level", result.scene_risk.risk_level)
                    metrics_col2.metric("Max Object Risk", f"{result.scene_risk.max_risk_score:.0f}")
                    metrics_col3.metric("Total Detections", result.detection_count)
                    
                    if result.scene_risk.reason:
                        st.info(result.scene_risk.reason)
                        
                    # Show Detections Table
                    if result.detections:
                        st.subheader("📋 Detected Objects")
                        df_data = []
                        for d in result.detections:
                            df_data.append({
                                "Class": d.class_name.upper(),
                                "Confidence": f"{d.confidence:.2f}",
                                "Risk Score": d.risk_score,
                                "In Danger Zone": "Yes" if d.in_danger_zone else "No",
                                "Reason": d.risk_reason,
                            })
                        df = pd.DataFrame(df_data).sort_values("Risk Score", ascending=False)
                        st.dataframe(df)
                        
                    # ── Image Downloads ──
                    st.subheader("💾 Export Results")
                    dl_col1, dl_col2, dl_col3 = st.columns(3)
                    
                    # 1. Image
                    img_bytes = export_image_to_bytes(annotated_bgr, ext=".png")
                    img_name = generate_export_filename("annotated", result.scene_risk.risk_level, ".png")
                    dl_col1.download_button(
                        label="📷 Download Image",
                        data=img_bytes,
                        file_name=img_name,
                        mime="image/png",
                    )
                    
                    # 2. JSON
                    json_str = export_report_to_json(result)
                    json_name = generate_export_filename("report", result.scene_risk.risk_level, ".json")
                    dl_col2.download_button(
                        label="📄 Download JSON",
                        data=json_str,
                        file_name=json_name,
                        mime="application/json",
                    )
                    
                    # 3. CSV
                    csv_str = export_table_to_csv(result)
                    csv_name = generate_export_filename("detections", result.scene_risk.risk_level, ".csv")
                    dl_col3.download_button(
                        label="📊 Download CSV",
                        data=csv_str,
                        file_name=csv_name,
                        mime="text/csv",
                    )
                        
                    # VIDEO ANALYSIS
                    output_video_path = tmp_path + "_annotated.mp4"
                    result = analyze_video(
                        tmp_path,
                        stride=stride,
                        model_name=model_name,
                        confidence_threshold=conf_thresh,
                        danger_zone_params=dz_params,
                        output_video_path=output_video_path,
                        use_lane_detection=use_lane_detection,
                        use_depth=use_depth,
                    )
                    
                    # Show Summary Metrics
                    st.subheader("📊 Video Analysis Summary")
                    m1, m2, m3, m4 = st.columns(4)
                    
                    # Overall risk is max across all frames
                    overall_risk_level = "LOW"
                    if result.max_scene_risk >= 70:
                        overall_risk_level = "HIGH"
                    elif result.max_scene_risk >= 35:
                        overall_risk_level = "MEDIUM"
                        
                    m1.metric("Overall Peak Risk", overall_risk_level)
                    m2.metric("Max Score", f"{result.max_scene_risk:.0f}")
                    m3.metric("High Risk Frames", result.high_risk_frames)
                    m4.metric("Avg Objects/Frame", result.avg_object_count)
                    
                    st.success(
                        f"Processed {result.frames_processed} frames out of {result.total_frames_read} "
                        f"(stride: {result.frame_stride})."
                    )
                    
                    # Extract and show the riskiest frame
                    if result.riskiest_frame_index >= 0:
                        st.subheader("📸 Riskiest Frame")
                        
                        # Find the frame result
                        riskiest_frame_res = next(
                            (fr for fr in result.frame_results if fr.frame_index == result.riskiest_frame_index),
                            None
                        )
                        
                        if riskiest_frame_res:
                            frame_bgr = get_video_frame(tmp_path, result.riskiest_frame_index)
                            if frame_bgr is not None:
                                # We need to construct a dummy AnalysisResult just for the annotator
                                from src.pipeline import AnalysisResult
                                from src.risk.scene_classifier import classify_scene
                                
                                dummy_scene_risk = classify_scene(riskiest_frame_res.detections)
                                dummy_res = AnalysisResult(
                                    image_width=frame_bgr.shape[1],
                                    image_height=frame_bgr.shape[0],
                                    channels=3,
                                    detections=riskiest_frame_res.detections,
                                    detection_count=riskiest_frame_res.detection_count,
                                    scene_risk=dummy_scene_risk,
                                    timestamp="",
                                    settings={}
                                )
                                
                                ann_bgr = annotate_image(frame_bgr, dummy_res, draw_danger_zone=draw_zone)
                                ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
                                
                                st.image(
                                    ann_rgb, 
                                    caption=f"Frame {result.riskiest_frame_index} (Risk: {dummy_scene_risk.risk_level})", 
                                    use_container_width=True
                                )
                                st.info(dummy_scene_risk.reason)
                            else:
                                st.warning("Could not extract frame from video.")

                    # ── Video Downloads ──
                    st.subheader("💾 Export Results")
                    dl_col1, dl_col2, dl_col3 = st.columns(3)
                    
                    # 1. JSON
                    json_str = export_report_to_json(result)
                    json_name = generate_export_filename("video_report", overall_risk_level, ".json")
                    dl_col1.download_button(
                        label="📄 Download JSON",
                        data=json_str,
                        file_name=json_name,
                        mime="application/json",
                    )
                    
                    # 2. CSV
                    csv_str = export_table_to_csv(result)
                    csv_name = generate_export_filename("video_detections", overall_risk_level, ".csv")
                    dl_col2.download_button(
                        label="📊 Download CSV",
                        data=csv_str,
                        file_name=csv_name,
                        mime="text/csv",
                    )
                    
                    # 3. Annotated Video
                    if result.annotated_video_path and os.path.exists(result.annotated_video_path):
                        with open(result.annotated_video_path, "rb") as f:
                            video_bytes = f.read()
                        
                        video_name = generate_export_filename("video_annotated", overall_risk_level, ".mp4")
                        dl_col3.download_button(
                            label="🎥 Download Video",
                            data=video_bytes,
                            file_name=video_name,
                            mime="video/mp4",
                        )
                        st.subheader("▶️ Play Annotated Video")
                        st.video(video_bytes)

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                
            finally:
                # Cleanup temp files
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                if 'output_video_path' in locals() and os.path.exists(output_video_path):
                    os.remove(output_video_path)
