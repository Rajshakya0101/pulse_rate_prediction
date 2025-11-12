import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import torch
import tempfile
import time
from pathlib import Path
from collections import deque

from tiny_rppgnet import TinyRPPGNet
from rppg_core import bandpass_filter, bpm_from_welch_harmonic

# Page configuration
st.set_page_config(
    page_title="Pulse Rate Monitor - Cloud",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive UI
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    div.stButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border: none;
        border-radius: 30px;
        padding: 15px 40px;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.4);
    }
    h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.15);
        padding: 15px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_resource
def load_model():
    """Load the TinyRPPGNet model (cached)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyRPPGNet().to(device)
    try:
        state = torch.load("models/tiny_rppgnet_best.pth", map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def get_forehead_bbox_from_landmarks(landmarks, w, h, scale=1.05, frac=0.35):
    """Extract forehead bounding box from face landmarks"""
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]

    x1 = max(int(min(xs) * w), 0)
    x2 = min(int(max(xs) * w), w - 1)
    y1 = max(int(min(ys) * h), 0)
    y2 = min(int(max(ys) * h), h - 1)

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale

    x1 = int(max(cx - bw / 2, 0))
    x2 = int(min(cx + bw / 2, w - 1))
    y1 = int(max(cy - bh / 2, 0))
    y2 = int(min(cy + bh / 2, h - 1))

    h_box = y2 - y1
    y2_new = y1 + int(h_box * frac)
    return x1, y1, x2, max(y1 + 1, y2_new)

def process_video(video_path, progress_bar, status_text):
    """Process uploaded video for heart rate detection"""
    
    model, device = load_model()
    if model is None:
        return None, None, None, None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Could not open video file.")
        return None, None, None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    roi_size = (36, 36)
    clip_len = 128
    roi_buffer = deque(maxlen=clip_len)
    trace_buffer = deque(maxlen=clip_len * 2)
    
    mp_face_mesh = mp.solutions.face_mesh
    
    frame_count = 0
    sample_frames = []
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            
            results = face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0]
                x1, y1, x2, y2 = get_forehead_bbox_from_landmarks(lms, w, h)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w - 1, x2); y2 = min(h - 1, y2)
                
                if x2 > x1 and y2 > y1:
                    roi = frame_rgb[y1:y2, x1:x2]
                    roi_resized = cv2.resize(roi, roi_size)
                    
                    roi_buffer.append(roi_resized)
                    
                    g_val = roi_resized[:, :, 1].mean()
                    trace_buffer.append(g_val)
                    
                    # Save sample frames with overlay
                    if frame_count % 30 == 0:  # Save every 30th frame
                        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        sample_frames.append(frame_bgr)
            
            # Update progress
            frame_count += 1
            if total_frames > 0:
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"⏱️ Processing: Frame {frame_count}/{total_frames}")
    
    cap.release()
    
    # Calculate heart rates
    bpm_classic = float("nan")
    if len(trace_buffer) > 64:
        try:
            trace = np.array(list(trace_buffer), dtype=np.float32)
            trace_f = bandpass_filter(trace, fs=fps, low=0.7, high=4.0)
            bpm_classic = bpm_from_welch_harmonic(trace_f, fs=fps)
        except:
            pass
    
    bpm_tiny = float("nan")
    if len(roi_buffer) >= clip_len:
        try:
            clip = np.stack(list(roi_buffer)[-clip_len:], axis=0)
            x = torch.from_numpy(clip).float() / 255.0
            x = x.permute(3, 0, 1, 2).unsqueeze(0).to(device)
            with torch.no_grad():
                bpm_tiny = model(x).cpu().item()
        except:
            pass
    
    return bpm_classic, bpm_tiny, list(trace_buffer), sample_frames

# Main UI
st.title("❤️ Pulse Rate Monitor")
st.markdown("### Heart Rate Detection from Video using rPPG Technology")

# Sidebar
with st.sidebar:
    st.header("⚙️ About")
    st.markdown("---")
    
    st.markdown("### 📊 How it works")
    st.info("""
    1. **Upload** a video of your face
    2. Algorithm detects your **forehead**
    3. Extracts **rPPG signals**
    4. Calculates heart rate using:
       - Classical signal processing
       - TinyRPPGNet deep learning
    """)
    
    st.markdown("---")
    st.markdown("### 🔬 Technology")
    st.success("""
    - **MediaPipe**: Face detection
    - **Classical Method**: FFT & Welch PSD
    - **TinyRPPGNet**: 3D CNN model
    - **rPPG**: Remote photoplethysmography
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Video Requirements")
    st.warning("""
    - **Duration**: 5-30 seconds
    - **Lighting**: Good, even lighting
    - **Position**: Face clearly visible
    - **Movement**: Keep face still
    - **Focus**: Forehead visible
    - **Format**: MP4, AVI, MOV, MKV
    """)
    
    st.markdown("---")
    st.markdown("### 📋 Normal Heart Rate")
    st.info("""
    - **Resting**: 60-100 BPM
    - **Athlete**: 40-60 BPM
    - **Exercise**: 100-150 BPM
    """)

# Main content
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader(
    "📹 Upload a video of your face",
    type=["mp4", "avi", "mov", "mkv", "webm"],
    help="Upload a short video (5-30 seconds) with your face clearly visible"
)

if uploaded_file is not None:
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    st.success(f"✅ Video uploaded: {uploaded_file.name}")
    
    # Show video preview
    st.video(uploaded_file)
    
    # Process button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Analyze Heart Rate", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🔄 Processing video..."):
                bpm_classic, bpm_tiny, trace, sample_frames = process_video(
                    tmp_path, progress_bar, status_text
                )
            
            progress_bar.empty()
            status_text.empty()
            
            # Clean up temp file
            try:
                Path(tmp_path).unlink()
            except:
                pass
            
            if bpm_classic is not None or bpm_tiny is not None:
                st.success("✅ Analysis completed!")
                
                # Display results
                st.markdown("---")
                st.markdown("### 📈 Heart Rate Results")
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    if np.isfinite(bpm_classic):
                        st.metric("🔵 Classical Method", f"{bpm_classic:.1f} BPM")
                    else:
                        st.warning("⚠️ Classical method: No valid reading")
                
                with result_col2:
                    if np.isfinite(bpm_tiny):
                        st.metric("🟣 TinyRPPGNet", f"{bpm_tiny:.1f} BPM")
                    else:
                        st.warning("⚠️ TinyRPPGNet: No valid reading")
                
                # Show sample frames
                if sample_frames:
                    st.markdown("---")
                    st.markdown("### 🖼️ Sample Frames with Detection")
                    cols = st.columns(min(len(sample_frames), 4))
                    for idx, frame in enumerate(sample_frames[:4]):
                        with cols[idx]:
                            st.image(frame, channels="BGR", use_container_width=True)
                
                # Show signal plot
                if trace and len(trace) > 0:
                    st.markdown("---")
                    st.markdown("### 📊 rPPG Signal Trace")
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(12, 4))
                    trace_array = np.array(trace)
                    trace_norm = (trace_array - np.mean(trace_array)) / (np.std(trace_array) + 1e-9)
                    ax.plot(trace_norm, color='#667eea', linewidth=1.5)
                    ax.set_xlabel("Frame", fontsize=12)
                    ax.set_ylabel("Normalized Amplitude", fontsize=12)
                    ax.set_title("rPPG Signal (Green Channel)", fontsize=14)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                
                # Health interpretation
                st.markdown("---")
                st.markdown("### 💡 Interpretation")
                
                avg_bpm = np.nanmean([
                    bpm_classic if np.isfinite(bpm_classic) else np.nan,
                    bpm_tiny if np.isfinite(bpm_tiny) else np.nan
                ])
                
                if np.isfinite(avg_bpm):
                    if 60 <= avg_bpm <= 100:
                        st.success(f"✅ **Normal resting heart rate**: {avg_bpm:.1f} BPM")
                    elif 100 < avg_bpm <= 120:
                        st.warning(f"⚠️ **Elevated heart rate**: {avg_bpm:.1f} BPM")
                    elif avg_bpm > 120:
                        st.error(f"🚨 **High heart rate**: {avg_bpm:.1f} BPM")
                    else:
                        st.info(f"💙 **Low heart rate**: {avg_bpm:.1f} BPM (May be normal for athletes)")
                else:
                    st.error("❌ Could not detect a valid heart rate. Please ensure:")
                    st.markdown("""
                    - Video has good lighting
                    - Your face is clearly visible
                    - You remain relatively still
                    - Video is at least 5 seconds long
                    """)
            else:
                st.error("❌ Analysis failed. Please try again with a different video.")

else:
    # Show instructions when no file uploaded
    st.info("""
    👆 **Upload a video to get started!**
    
    📱 **How to record a good video:**
    1. Find a well-lit area
    2. Position camera to show your face clearly
    3. Keep still for 10-15 seconds
    4. Ensure your forehead is visible
    5. Upload the video above
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <p>🏥 <b>Disclaimer:</b> This is for educational purposes only. Not for medical diagnosis.</p>
    <p>💻 Powered by TinyRPPGNet, MediaPipe & PyTorch</p>
    <p>🎓 Remote Photoplethysmography (rPPG) Demo</p>
</div>
""", unsafe_allow_html=True)
