import streamlit as st
import cv2
import numpy as np
import torch
import tempfile
import time
from pathlib import Path
from collections import deque

# Optional mediapipe import (safe fallback)
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

from tiny_rppgnet import TinyRPPGNet
from rppg_core import bandpass_filter, bpm_from_welch_harmonic

# Page configuration
st.set_page_config(
    page_title="Pulse Rate Monitor - Cloud",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main, .stApp {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
div.stButton > button {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  color: white; font-size: 18px; font-weight: bold;
  border: none; border-radius: 30px;
  padding: 15px 40px; box-shadow: 0 8px 15px rgba(0, 0, 0, 0.3);
  transition: all 0.3s ease;
}
div.stButton > button:hover {
  transform: translateY(-3px);
  box-shadow: 0 12px 20px rgba(0, 0, 0, 0.4);
}
h1 { color: white; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); }
.stMetric {
  background: rgba(255, 255, 255, 0.15);
  padding: 15px; border-radius: 15px; backdrop-filter: blur(10px);
}
</style>
""", unsafe_allow_html=True)

# Session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_resource
def load_model():
    """Load TinyRPPGNet"""
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
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    x1, x2 = int(min(xs)*w), int(max(xs)*w)
    y1, y2 = int(min(ys)*h), int(max(ys)*h)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = (x2-x1)*scale, (y2-y1)*scale
    x1, x2 = int(max(cx-bw/2,0)), int(min(cx+bw/2,w-1))
    y1, y2 = int(max(cy-bh/2,0)), int(min(cy+bh/2,h-1))
    y2 = y1 + int((y2-y1)*frac)
    return x1, y1, x2, max(y1+1, y2)


def process_video(video_path, progress_bar, status_text):
    model, device = load_model()
    if model is None:
        return None, None, None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Could not open video file.")
        return None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    roi_size = (36, 36)
    clip_len = 128
    roi_buffer, trace_buffer = deque(maxlen=clip_len), deque(maxlen=clip_len*2)
    frame_count, sample_frames = 0, []

    if MP_AVAILABLE:
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while True:
                ok, frame_bgr = cap.read()
                if not ok: break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                h, w = frame_rgb.shape[:2]
                res = face_mesh.process(frame_rgb)
                if res.multi_face_landmarks:
                    lms = res.multi_face_landmarks[0]
                    x1,y1,x2,y2 = get_forehead_bbox_from_landmarks(lms,w,h)
                    roi = frame_rgb[y1:y2,x1:x2]
                    if roi.size:
                        roi_resized = cv2.resize(roi, roi_size)
                        roi_buffer.append(roi_resized)
                        trace_buffer.append(roi_resized[:,:,1].mean())
                        if frame_count%30==0:
                            cv2.rectangle(frame_bgr,(x1,y1),(x2,y2),(0,255,0),2)
                            sample_frames.append(frame_bgr)
                frame_count+=1
                if total_frames>0:
                    progress=min(frame_count/total_frames,1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"⏱️ Processing: Frame {frame_count}/{total_frames}")
    else:
        face_det=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
        while True:
            ok,frame_bgr=cap.read()
            if not ok: break
            gray=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2GRAY)
            faces=face_det.detectMultiScale(gray,1.2,5)
            if len(faces):
                x,y,wf,hf=max(faces,key=lambda r:r[2]*r[3])
                x1,y1,x2,y2=x,y,x+wf,y+int(0.3*hf)
                roi=frame_bgr[y1:y2,x1:x2]
                if roi.size:
                    roi_resized=cv2.resize(roi,roi_size)
                    roi_buffer.append(roi_resized)
                    trace_buffer.append(roi_resized[:,:,1].mean())
            frame_count+=1
            if total_frames>0:
                progress=min(frame_count/total_frames,1.0)
                progress_bar.progress(progress)
                status_text.text(f"⏱️ Processing: Frame {frame_count}/{total_frames}")

    cap.release()
    bpm_classic,bpm_tiny=float("nan"),float("nan")
    if len(trace_buffer)>64:
        trace=np.array(trace_buffer,dtype=np.float32)
        trace_f=bandpass_filter(trace,fs=fps,low=0.7,high=4.0)
        bpm_classic=bpm_from_welch_harmonic(trace_f,fs=fps)
    if len(roi_buffer)>=clip_len:
        clip=np.stack(list(roi_buffer)[-clip_len:],axis=0)
        x=torch.from_numpy(clip).float()/255.0
        x=x.permute(3,0,1,2).unsqueeze(0).to(device)
        with torch.no_grad(): bpm_tiny=model(x).cpu().item()
    return bpm_classic,bpm_tiny,list(trace_buffer),sample_frames


# --- UI ---
st.title("❤️ Pulse Rate Monitor")
st.markdown("### Heart Rate Detection from Video using rPPG Technology")

uploaded_file=st.file_uploader("📹 Upload a face video",
    type=["mp4","avi","mov","mkv","webm"],
    help="Upload a short video (5–30 s) with your face clearly visible")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False,suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path=tmp.name
    st.video(uploaded_file)
    if st.button("🔍 Analyze Heart Rate", use_container_width=True):
        progress_bar=st.progress(0); status_text=st.empty()
        with st.spinner("🔄 Processing video..."):
            bpm_c,bpm_t,trace,frames=process_video(tmp_path,progress_bar,status_text)
        progress_bar.empty(); status_text.empty()
        Path(tmp_path).unlink(missing_ok=True)
        st.success("✅ Analysis completed!")
        c1,c2=st.columns(2)
        c1.metric("🔵 Classical", f"{bpm_c:.1f} BPM" if np.isfinite(bpm_c) else "—")
        c2.metric("🟣 TinyRPPGNet", f"{bpm_t:.1f} BPM" if np.isfinite(bpm_t) else "—")

else:
    st.info("👆 Upload a short, well-lit face video to get started.")
