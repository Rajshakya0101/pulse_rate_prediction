# 🚀 Streamlit Cloud Deployment Guide

## 📋 Prerequisites

✅ Your code is already pushed to GitHub
✅ Repository: `Rajshakya0101/pulse_rate_prediction`
✅ All required files are in place:
- `streamlit_app.py` (main app file)
- `requirements.txt` (Python dependencies)
- `packages.txt` (system dependencies)
- `models/tiny_rppgnet_best.pth` (model file)
- `.streamlit/config.toml` (configuration)

## 🎯 Deployment Steps

### Step 1: Go to Streamlit Cloud
1. Visit: https://share.streamlit.io/
2. Click **"Sign in with GitHub"**
3. Authorize Streamlit to access your GitHub account

### Step 2: Deploy New App
1. Click **"New app"** button
2. Fill in the following details:
   - **Repository**: `Rajshakya0101/pulse_rate_prediction`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **App URL** (optional): Choose a custom URL or use default

### Step 3: Advanced Settings (Optional)
Click **"Advanced settings"** if you need to:
- Set Python version: `3.11` (recommended)
- Add secrets (not needed for this app)
- Configure resources

### Step 4: Deploy!
1. Click **"Deploy!"** button
2. Wait for deployment (5-10 minutes)
3. Streamlit will:
   - Clone your repository
   - Install system packages from `packages.txt`
   - Install Python packages from `requirements.txt`
   - Launch your app

## ⚠️ Important Notes

### Webcam Access Limitation
**Streamlit Cloud apps cannot access user webcams directly** due to security restrictions in the cloud environment.

### Solutions:

#### Option 1: Local Deployment Only
- Keep the live webcam version for **local use only**
- Users run: `streamlit run streamlit_app.py` on their machines

#### Option 2: Upload Video Version (Recommended for Cloud)
Create an alternative version that accepts **uploaded video files** instead of live webcam. I can help you create this version if needed.

#### Option 3: WebRTC Integration
Use `streamlit-webrtc` component for browser-based webcam access (more complex setup).

## 📱 What Works on Streamlit Cloud

✅ UI and styling
✅ Model loading
✅ Signal processing algorithms
✅ Video file upload and processing
❌ Direct webcam access (requires local deployment)

## 🔄 Alternative: Deploy with Video Upload

Would you like me to create a version that works on Streamlit Cloud by allowing users to **upload pre-recorded videos** instead of using live webcam?

This would include:
- File uploader for video files
- Same heart rate detection algorithms
- Same beautiful UI
- Works perfectly on Streamlit Cloud

## 🛠️ Post-Deployment

Once deployed, you can:
- Share your app URL with anyone
- Monitor app usage and logs
- Update by pushing to GitHub (auto-redeploys)
- Manage settings from Streamlit Cloud dashboard

## 📞 Need Help?

If you encounter issues:
1. Check Streamlit Cloud logs
2. Verify all dependencies in `requirements.txt`
3. Ensure model file is committed to Git
4. Check that file is < 100MB (Git LFS if larger)

## 🎉 Your App URL

After deployment, your app will be available at:
`https://[your-app-name].streamlit.app`

---

**Would you like me to create a video-upload version that works on Streamlit Cloud?**
