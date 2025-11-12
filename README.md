# ❤️ Live Pulse Rate Monitor

Real-time heart rate detection using Remote Photoplethysmography (rPPG) technology.

## 🎯 Features

- **Video Upload Processing**: Upload videos for heart rate analysis (Cloud-compatible)
- **Live Webcam Mode**: Real-time detection (Local deployment only)
- **Dual Detection Methods**:
  - Classical signal processing (FFT & Welch PSD)
  - Deep learning (TinyRPPGNet 3D CNN)
- **Beautiful UI**: Modern gradient design with real-time metrics
- **Forehead Detection**: Uses MediaPipe face mesh for accurate ROI extraction

## 🚀 Live Demo

Try it now: **[Launch App](https://your-app-name.streamlit.app)** *(Update after deployment)*

## 📂 Files

- `streamlit_app.py` - Main app (video upload, cloud-compatible)
- `streamlit_app_local.py` - Local webcam version
- `app.py` - OpenCV standalone version
- `tiny_rppgnet.py` - Deep learning model
- `rppg_core.py` - Signal processing utilities

## 🔬 Technology Stack

- **Streamlit**: Web application framework
- **OpenCV**: Video processing
- **MediaPipe**: Face detection and landmark extraction
- **PyTorch**: Deep learning model inference
- **SciPy**: Signal processing and filtering

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Rajshakya0101/pulse_rate_prediction.git
cd pulse_rate_prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

## 💻 Usage

### Cloud Version (Video Upload)
1. Visit the deployed app URL
2. Click **"Upload a video of your face"**
3. Select a video file (MP4, AVI, MOV, etc.)
4. Click **"Analyze Heart Rate"**
5. View results and signal visualization

### Local Version (Live Webcam)
```bash
streamlit run streamlit_app_local.py
```
1. Click **"Start Live Detection"** button
2. Position your face in front of the camera
3. Stay still for best results (5-10 seconds)
4. View real-time heart rate updates
5. Click **"Stop Now"** to end detection early

## 📊 How It Works

### rPPG (Remote Photoplethysmography)

The app detects subtle color changes in your forehead caused by blood flow:

1. **Face Detection**: MediaPipe locates facial landmarks
2. **ROI Extraction**: Forehead region is isolated
3. **Signal Processing**: Green channel intensity is tracked over time
4. **Filtering**: Band-pass filter (0.7-4.0 Hz / 42-240 BPM)
5. **HR Estimation**: 
   - Classical: FFT peak detection
   - Deep Learning: TinyRPPGNet CNN regression

## 🏗️ Model Architecture

**TinyRPPGNet**: Lightweight 3D CNN for video-based heart rate estimation
- Input: (Batch, 3 channels, 128 frames, 36×36 pixels)
- Output: BPM value (beats per minute)
- Architecture: 4 Conv3D layers + Global Average Pooling + 2 FC layers

## 📋 Requirements

- Python 3.8+
- Webcam access
- Good lighting conditions
- Stable internet (for cloud deployment)

## ⚠️ Disclaimer

This application is for **educational and demonstration purposes only**. It is not intended for medical diagnosis or clinical use. Always consult healthcare professionals for medical advice.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Rajshakya0101**
- GitHub: [@Rajshakya0101](https://github.com/Rajshakya0101)

## 🙏 Acknowledgments

- MediaPipe for face detection
- PyTorch community
- Streamlit for the amazing framework
