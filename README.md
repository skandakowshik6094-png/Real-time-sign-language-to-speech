# SignSense — Real-Time Sign Language Recognition

> Live sign-language detection in your browser, powered by **MediaPipe + PyTorch + Flask**.  
> Train new signs without leaving the UI — existing symbols are always preserved.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🖐 **2-Hand Detection** | Both hands tracked simultaneously via MediaPipe |
| 🧠 **Custom MLP Classifier** | 63-feature landmark → 2-layer BatchNorm MLP |
| ➕ **In-Browser Training** | Collect 60 samples, click Train — new symbol added live |
| 🔄 **Fix & Retrain All** | Full rebuild from scratch when needed |
| 🔒 **Safe Incremental Training** | Old symbol weights are bit-for-bit preserved |
| 📝 **Auto Sentence Builder** | Cooldown-gated, confidence-gated auto-append |
| 🔊 **Text-to-Speech** | Every detected sign is spoken aloud |
| 📷 **Multi-Camera Support** | Switch camera device from the UI |

---

## 🖼 Demo

![SignSense UI](Cheatsheet.png)

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager — `pip install uv`
- A webcam

### 1. Clone
```bash
git clone https://github.com/Arilsrinivas/sign.git
cd sign
```

### 2. Install dependencies
```bash
uv sync
```

### 3. Download MediaPipe hand model
```bash
python -c "
import urllib.request, os
os.makedirs('pretrained', exist_ok=True)
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
    'pretrained/hand_landmarker.task'
)"
```

### 4. Run
```bash
uv run src/app.py
```
Open **http://localhost:5000** in your browser.

---

## 🎯 How to Train a New Sign

1. Type the sign name in **"Add / Replace Symbol"** (e.g. `hello`)
2. Set **Samples = 60** (default)
3. Click **▶ Start Collecting** and perform the gesture in front of the camera
4. Wait for collection to complete (progress bar)
5. Click **🧠 Train Model** — the new symbol is added in ~30 seconds
6. Start Detection and test it!

> **🔄 Fix & Retrain All** — use this button if detection ever breaks. It rebuilds the model from scratch using all saved samples.

---

## 📁 Project Structure

```
sign/
├── src/
│   ├── app.py                 # Flask server + training routes
│   ├── detection_engine.py    # Camera loop, hand detection, classification
│   ├── hand_landmarks.py      # MediaPipe wrapper + EMA smoothing
│   ├── landmark_model.py      # 2-layer MLP (63 → 128 → 64 → N)
│   ├── landmark_trainer.py    # Dataset loader
│   ├── config.json            # Class names + camera config
│   └── templates/
│       └── index.html         # Full single-page UI
├── checkpoints/               # Trained model (.pt) — not in git
├── pretrained/                # MediaPipe .task file — not in git
├── pyproject.toml
└── README.md
```

---

## 🏗 Model Architecture

```
Input: 63 floats (21 landmarks × x,y,z — wrist-relative, normalised)
   ↓
FC1 (63 → 128) + BatchNorm + ReLU + Dropout(0.3)
   ↓
FC2 (128 → 64) + BatchNorm + ReLU + Dropout(0.3)
   ↓
FC3 (64 → N classes)  ← only this layer trains for new symbols
   ↓
Softmax → class + confidence %
```

**Safe incremental training:**  
When you add a new symbol, **only the single output neuron** for that class is trained.  
The backbone (FC1, FC2, BN layers) is frozen — old symbols are guaranteed unchanged.

---

## 🌐 Deployment

> ⚠️ **Note:** This app requires a **webcam/camera** connected to the machine running the server.  
> It cannot run as a pure cloud service — the camera must be physically available.

### Option A — Run on your own PC and share via ngrok
```bash
# Install ngrok from https://ngrok.com
ngrok http 5000
# Share the https URL with anyone
```

### Option B — Docker (self-hosted VPS with USB camera passthrough)
```bash
docker build -t signsense .
docker run --device=/dev/video0 -p 5000:5000 signsense
```

### Option C — Render / Railway (demo mode — no camera)
Deploy as a normal Flask app. The UI will load but detection requires  
a camera, so it works as a **demo / training interface** without live detection.

---

## 🛠 Tech Stack

- **MediaPipe** — hand landmark detection
- **PyTorch** — MLP classifier training & inference  
- **OpenCV** — camera capture & frame processing
- **Flask** — web server + SSE event streaming
- **pyttsx3** — text-to-speech
- **uv** — fast Python package manager

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Arilsrinivas**  
GitHub: [@Arilsrinivas](https://github.com/Arilsrinivas)
