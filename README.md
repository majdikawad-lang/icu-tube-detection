# 🫁 Autonomous ICU Airway Guardian Nexus

### *AI-Powered Endotracheal Tube Displacement Detection System*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/DICOM-Medical%20Grade-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  <i>⚕️ A production-ready, AI-powered clinical decision support system for real-time ETT displacement detection in ICU chest X-rays.</i>
</p>

-----

## 🚨 Clinical Significance

Endotracheal tube (ETT) displacement is a **life-threatening emergency** in intensive care units. A misplaced tube — too deep into a bronchus or too close to the vocal cords — can cause:

- 🫁 **Unilateral lung collapse** (endobronchial intubation)
- 🧠 **Hypoxic brain injury** within minutes
- ❤️ **Cardiac arrest** in critical cases

Traditionally, ICU nurses and physicians manually review chest X-rays to assess tube position — a process vulnerable to **human fatigue, cognitive overload, and delayed response**.

**The Autonomous ICU Airway Guardian Nexus** eliminates this gap by delivering **instant, AI-driven analysis** of DICOM chest X-rays, automatically measuring ETT-to-Carina distance and triggering **medical-grade alerts** when dangerous displacement is detected — empowering clinicians to act in seconds, not minutes.

-----

## 🧠 Project Overview

```
Upload DICOM ──► FastAPI Backend ──► Attention U-Net ──► Segmentation Mask
                                                              │
                                                              ▼
                                               Carina Landmark Detection
                                                              │
                                                              ▼
                                              Distance Calculation (cm)
                                                              │
                                          ┌───────────────────┴───────────────────┐
                                          ▼                                       ▼
                                   ✅ SAFE STATUS                        🚨 DISPLACEMENT ALERT
                              (Green Neon UI + Tone)              (Red Pulsing UI + Audio Alarm)
```

The system accepts raw `.dcm` DICOM files, processes them through a trained **Attention U-Net** deep learning model, calculates the precise ETT tip-to-Carina distance using pixel-to-millimeter DICOM metadata calibration, and returns a real-time annotated overlay with an audio-visual alert classification.

-----

## ✨ Features

|Feature                               |Description                                                                                               |
|--------------------------------------|----------------------------------------------------------------------------------------------------------|
|🤖 **Attention U-Net Inference**       |Fine-tuned on the 12GB RANZCR CLiP Kaggle dataset for high-precision ETT segmentation                     |
|🏥 **Medical-Grade DICOM Processing**  |Directly reads `.dcm` files, normalizes pixel arrays, and extracts spatial calibration metadata           |
|📏 **Pixel-to-mm Distance Calibration**|Uses DICOM `PixelSpacing` and `ImagerPixelSpacing` tags for clinically accurate measurements              |
|🎨 **AI Overlay Visualization**        |Blends predicted segmentation mask and anatomical landmarks directly over the original X-ray              |
|🔴🟢 **Audio-Visual Medical Alerts**    |Pulsing neon UI — Red for displacement danger, Green for safe — with synthetic Web Audio API warning beeps|
|💎 **Glassmorphism Dark UI**           |Stunning dark-mode frontend with CSS micro-animations and a futuristic clinical aesthetic                 |
|⚡ **Async FastAPI Backend**           |High-performance asynchronous API for low-latency inference serving                                       |
|🚀 **Zero-Framework Frontend**         |Pure HTML5, CSS3, and Vanilla JavaScript — highly optimized, no bloat                                     |

-----

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     FRONTEND (Browser)                  │
│  ┌─────────────┐    ┌──────────────┐   ┌─────────────┐  │
│  │  DICOM File │───►│  Fetch API   │   │  Web Audio  │  │
│  │   Upload    │    │  (POST /api) │   │     API     │  │
│  └─────────────┘    └──────┬───────┘   └──────▲──────┘  │
│                            │                  │          │
│                    ┌───────▼────────┐         │          │
│                    │  Result Render │─────────┘          │
│                    │  + AI Overlay  │  Trigger Alert     │
│                    └───────────────┘                     │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP POST (multipart/form-data)
┌──────────────────────────▼──────────────────────────────┐
│                  FASTAPI BACKEND (Uvicorn)               │
│                                                          │
│  1. Receive .dcm file                                    │
│  2. Pydicom → Extract pixel array + PixelSpacing        │
│  3. OpenCV → CLAHE + Bilateral Filter preprocessing     │
│  4. PyTorch → Attention U-Net inference                 │
│  5. Mask → Keypoint extraction (ETT tip + Carina)       │
│  6. Distance = pixels × PixelSpacing → cm              │
│  7. Return: overlay image + distance + status JSON      │
└─────────────────────────────────────────────────────────┘
```

-----

## 🛠️ Technology Stack

|Layer               |Technology                |Purpose                                           |
|--------------------|--------------------------|--------------------------------------------------|
|**Deep Learning**   |PyTorch + Attention U-Net |ETT segmentation and keypoint detection           |
|**Training Data**   |RANZCR CLiP (12GB, Kaggle)|Medical X-ray tube annotation dataset             |
|**Backend**         |FastAPI + Uvicorn         |Async API server and model inference              |
|**Image Processing**|OpenCV (cv2), NumPy       |Preprocessing, mask processing, overlay generation|
|**Medical Data**    |Pydicom                   |DICOM file parsing and metadata extraction        |
|**Frontend**        |HTML5, CSS3, Vanilla JS   |Glassmorphism UI, animations, Web Audio API alerts|

-----

## ⚙️ Prerequisites & Installation

### System Requirements

- Python **3.9+**
- CUDA-compatible GPU *(recommended for real-time inference)*
- 8GB+ RAM

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/icu-tube-detection.git
cd icu-tube-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```
fastapi==0.104.1
uvicorn==0.24.0
torch==2.1.0
torchvision==0.16.0
opencv-python==4.8.1.78
numpy==1.24.4
pydicom==2.4.3
python-multipart==0.0.6
Pillow==10.1.0
```

### 4. Add Your Trained Model

Place your trained Attention U-Net weights file in the `models/` directory:

```
icu-tube-detection/
├── models/
│   └── attention_unet_ranzcr.pth   ← your model weights here
├── static/
├── main.py
└── requirements.txt
```

### 5. Run the Application

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at:

```
http://localhost:8000
```

-----

## 🖥️ Usage Guide

### Step 1 — Upload a DICOM File

- Click the **upload zone** or drag and drop a `.dcm` chest X-ray file
- The system accepts standard DICOM format from any ICU imaging modality

### Step 2 — AI Analysis

- The backend automatically:
  - Preprocesses the X-ray (CLAHE contrast enhancement + bilateral denoising)
  - Runs Attention U-Net inference for ETT segmentation
  - Detects the ETT tip and Carina landmark keypoints
  - Calculates the precise distance in **centimeters**

### Step 3 — Read the Result

|Result                              |UI State              |Audio                 |
|------------------------------------|----------------------|----------------------|
|✅ **SAFE** — ETT within normal range|🟢 Green neon glow     |Steady safe tone      |
|🚨 **DANGER** — ETT displaced        |🔴 Pulsing red alarm UI|Repeating medical beep|

### Step 4 — Review the AI Overlay

- The annotated X-ray is displayed with:
  - 🔵 **Segmentation mask** overlaid on the tube
  - 📍 **ETT tip marker**
  - 📍 **Carina landmark marker**
  - 📏 **Distance measurement label in cm**

-----

## 🔬 Model Details

|Property        |Value                               |
|----------------|------------------------------------|
|Architecture    |Attention U-Net                     |
|Training Dataset|RANZCR CLiP (Kaggle)                |
|Dataset Size    |~12 GB, 30,000+ annotated X-rays    |
|Input Size      |512 × 512 px                        |
|Output          |Binary segmentation mask            |
|Framework       |PyTorch 2.0                         |
|Preprocessing   |CLAHE + Bilateral Filtering (OpenCV)|

-----

## 🚀 Future Improvements

- [ ] 🧠 **Multi-line Detection** — Extend model to detect NGT, CVCs, and chest drains simultaneously
- [ ] 📊 **Trend Monitoring Dashboard** — Track ETT position changes across serial X-rays over time
- [ ] 🔗 **PACS/HL7 Integration** — Direct integration with hospital Picture Archiving Systems
- [ ] 📱 **Mobile-Responsive UI** — Tablet-optimized interface for bedside clinical use
- [ ] ☁️ **Cloud Deployment** — Docker containerization and Azure/AWS deployment pipeline
- [ ] 🏋️ **Model Distillation** — Lightweight model variant for edge deployment on ICU workstations
- [ ] 🔐 **HIPAA Compliance Layer** — Patient data encryption and audit logging

-----

## 👨‍💻 Author

**Majdi Awad**
*AI-Powered Web Developer | Computer Science Graduate*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/majdi-awad)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/majdi-awad)

-----

## ⚠️ Medical Disclaimer

> This system is intended as a **clinical decision support tool** and does not replace the judgment of qualified medical professionals. All AI-generated results must be reviewed and confirmed by a licensed physician or radiologist before any clinical action is taken.

-----

<p align="center">
  <i>Built with ❤️ for the ICU — where every second counts.</i>
</p>
