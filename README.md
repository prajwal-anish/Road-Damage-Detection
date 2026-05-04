# 🚗 Road Surface Damage Detection System

**Detect Potholes, Cracks & Manholes using YOLOv8 + FastAPI + React**

[![Model](https://img.shields.io/badge/Model-YOLOv8s-blue)](https://github.com/ultralytics/ultralytics)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Frontend](https://img.shields.io/badge/Frontend-React%2018-61DAFB)](https://react.dev/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF)](https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes)

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PART 1 — MODEL TRAINING (Colab)                   │
│                                                                       │
│  Kaggle Dataset ──► Data Prep ──► YOLOv8s Training ──► best.pt      │
│  (YOLO format)     (75/15/10)    (100 epochs, GPU)    (exported)     │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │ best.pt
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PART 2 — APPLICATION (Local)                      │
│                                                                       │
│  ┌─────────────────────┐        ┌──────────────────────────────┐    │
│  │  React Frontend     │  HTTP  │  FastAPI Backend              │    │
│  │  (Vite + React 18)  │◄──────►│  POST /predict               │    │
│  │                     │        │  ├── Load YOLOv8 model        │    │
│  │  • Image Upload     │        │  ├── Run inference            │    │
│  │  • Canvas Boxes     │        │  ├── Compute severity         │    │
│  │  • Severity Cards   │        │  ├── Log detections           │    │
│  │  • Confidence Tune  │        │  └── Return JSON + image      │    │
│  └─────────────────────┘        └──────────────────────────────┘    │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  OpenCV Real-Time Detection (realtime_detection.py)           │   │
│  │  Webcam ──► YOLOv8 Inference ──► Live bounding boxes + HUD   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
road_damage_detection/
│
├── colab_notebook/
│   └── road_damage_training.py     # Full Colab training script
│
├── backend/
│   ├── main.py                     # FastAPI application
│   ├── realtime_detection.py       # OpenCV webcam detection
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .env.example
│   └── utils/
│       ├── severity.py             # Damage severity scoring
│       └── logger.py               # Structured logging
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── .env.example
│   └── src/
│       ├── main.jsx
│       └── App.jsx                 # Complete React UI
│
├── model/
│   └── best.pt                     # ← Place your trained model here
│
└── README.md
```

---

## 🏋️ Part 1: Model Training (Google Colab)

### Step 1 — Open Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook** and upload `colab_notebook/road_damage_training.py`
   - Or: New notebook → paste cell by cell (each `# %%` marks a new cell)
3. Set Runtime: **Runtime → Change runtime type → T4 GPU**

### Step 2 — Get Kaggle API Key

1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
2. Scroll to **API** section → Click **Create New Token**
3. Download `kaggle.json`

### Step 3 — Run all cells in order

The notebook will:
1. Install dependencies (ultralytics, kaggle, opencv)
2. Mount your Google Drive
3. Upload `kaggle.json` and download the dataset
4. Validate and split dataset (75/15/10)
5. Visualize class distribution and sample annotations
6. Train YOLOv8s for 100 epochs with augmentations
7. Plot training curves and confusion matrix
8. Run validation and display test predictions
9. Save `best.pt` to Google Drive + offer local download

### Expected Training Time

| GPU         | ~Time    |
|-------------|----------|
| T4 (free)   | 40–60min |
| A100        | 10–20min |
| V100        | 20–35min |

### Expected Metrics (approx.)

| Metric       | Target  |
|--------------|---------|
| mAP@50       | ~0.65+  |
| mAP@50-95    | ~0.40+  |
| Precision    | ~0.70+  |
| Recall       | ~0.60+  |

---

## 💻 Part 2: Local Application

### Prerequisites

- Python 3.10+
- Node.js 18+
- `best.pt` model file (from Colab)

---

### Backend Setup

```bash
# 1. Navigate to backend
cd road_damage_detection/backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — set MODEL_PATH to ../model/best.pt

# 5. Place your trained model
cp /path/to/best.pt ../model/best.pt

# 6. Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health: http://localhost:8000/health

---

### Frontend Setup

```bash
# 1. Navigate to frontend
cd road_damage_detection/frontend

# 2. Install dependencies
npm install

# 3. Configure environment
cp .env.example .env
# .env: VITE_API_URL=http://localhost:8000

# 4. Start development server
npm run dev
```

Frontend will be at: **http://localhost:5173**

---

### Real-Time Webcam Detection

```bash
cd backend

# Webcam
python realtime_detection.py --source 0 --conf 0.3

# Video file
python realtime_detection.py --source road_video.mp4 --conf 0.25

# Custom model
python realtime_detection.py --source 0 --model ../model/best.pt --conf 0.3
```

**Controls:**
- `Q` or `ESC` — Quit
- `+` / `-` — Adjust confidence threshold live
- `S` — Save screenshot
- `P` — Pause/resume

---

## 🔌 API Reference

### `POST /predict`

Detect road damage in an uploaded image.

**Request:**
```
Content-Type: multipart/form-data
file: <image file>
conf: 0.25          (optional, confidence threshold)
iou: 0.45           (optional, IoU threshold)
include_annotated_image: true  (optional)
```

**Response:**
```json
{
  "request_id": "a1b2c3d4",
  "timestamp": "2024-01-15T12:30:00Z",
  "image_width": 1920,
  "image_height": 1080,
  "detections": [
    {
      "id": 0,
      "class_id": 0,
      "class_name": "pothole",
      "class_emoji": "🕳️",
      "confidence": 0.8721,
      "bbox": {
        "x1": 245.2, "y1": 388.5,
        "x2": 512.8, "y2": 621.3,
        "width": 267.6, "height": 232.8,
        "center_x": 379.0, "center_y": 504.9
      },
      "area_px": 62338.88,
      "area_ratio": 0.030,
      "severity": "HIGH",
      "severity_score": 0.6214,
      "color": "#E74C3C"
    }
  ],
  "summary": {
    "total_detections": 3,
    "class_counts": {"pothole": 2, "crack": 1},
    "overall_severity": "HIGH",
    "overall_severity_score": 0.6214,
    "dominant_class": "pothole",
    "affected_area_ratio": 0.085
  },
  "inference_time_ms": 42.5,
  "model_confidence_threshold": 0.25
}
```

### `GET /health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "../model/best.pt",
  "total_requests": 142,
  "avg_inference_ms": 38.4,
  "uptime_seconds": 3600.0
}
```

### `GET /stats`

```json
{
  "total_inferences": 142,
  "avg_inference_ms": 38.4,
  "class_detections": {
    "pothole": 87,
    "crack": 43,
    "manhole": 12
  }
}
```

### `POST /predict/batch`

Upload up to 10 images for batch detection.

---

## ⚙️ Bonus Features

### 1. Damage Severity Scoring

The system uses a multi-factor severity algorithm:

```
severity_score = (0.5 × confidence + 0.5 × normalized_area) × class_weight

Class weights:
  pothole  → 1.00  (highest danger)
  crack    → 0.75  (moderate)
  manhole  → 0.40  (infrastructure)

Severity levels:
  0.00 – 0.25  → LOW      (green)
  0.25 – 0.50  → MEDIUM   (orange)
  0.50 – 0.75  → HIGH     (dark orange)
  0.75 – 1.00  → CRITICAL (red)
```

### 2. Structured Logging System

Every detection is logged in two formats:
- **Text log:** `backend/logs/api.log` (rotated at 10MB)
- **JSON log:** `backend/logs/detections.jsonl` (append-only JSONL)

Sample JSONL entry:
```json
{
  "event": "detection",
  "request_id": "a1b2c3d4",
  "timestamp": "2024-01-15T12:30:00Z",
  "filename": "road_image.jpg",
  "num_detections": 3,
  "class_counts": {"pothole": 2, "crack": 1},
  "inference_ms": 42.5,
  "severity": "HIGH"
}
```

### 3. Confidence Threshold Tuning

Adjust confidence live:
- **Frontend:** Slider control (0.05 – 0.95)
- **API:** `?conf=0.35` query parameter
- **Realtime:** `+` / `-` keyboard shortcuts

### 4. Performance Optimization

Realtime detection uses frame skipping:
```python
SKIP_FRAMES = 1  # Run inference every 2nd frame
# Reduces GPU load while maintaining visual smoothness
```

Backend optimization:
- Model preloaded on startup (no cold start)
- `augment=False` in inference for speed
- Fixed `imgsz=640` for consistent performance

---

## 🚀 Deployment

### Backend → Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
cd backend
railway init
railway up
```

Set environment variables in Railway dashboard:
```
MODEL_PATH=./model/best.pt
CONF_THRESHOLD=0.25
```

**Important:** Upload `best.pt` to the Railway deployment:
- Add `model/best.pt` to your repository (git-lfs for large files), OR
- Use Railway volumes

### Backend → Render

1. Connect GitHub repo to Render
2. Set Build Command: `pip install -r requirements.txt`
3. Set Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables
5. Upload model via Render's disk feature

### Frontend → Vercel

```bash
cd frontend
npm run build

# Using Vercel CLI
npx vercel --prod
```

Or connect GitHub to Vercel dashboard and set:
- Framework: Vite
- Build command: `npm run build`
- Output dir: `dist`
- Environment variable: `VITE_API_URL=https://your-backend.railway.app`

---

## 🗂️ Dataset Details

| Property    | Value                             |
|-------------|-----------------------------------|
| Source      | Kaggle (Lorenzo Arcioni)          |
| Format      | YOLO (normalized bounding boxes)  |
| Classes     | 0=Pothole, 1=Crack, 2=Manhole     |
| Train split | 75%                               |
| Val split   | 15%                               |
| Test split  | 10%                               |

---

## 🔧 Troubleshooting

**`Model file not found`**
```bash
ls model/best.pt  # Check if file exists
# If not, download from Google Drive after training
```

**`CUDA out of memory` in Colab**
```python
# Reduce batch size in training config:
TRAINING_CONFIG['batch'] = 8  # Default is 16
```

**`CORS error` in browser**
```python
# backend/main.py already has CORS middleware for all origins
# For production, restrict to your frontend domain:
allow_origins=["https://your-app.vercel.app"]
```

**`Cannot open webcam` (realtime)**
```bash
# List available cameras
python -c "import cv2; [print(i, cv2.VideoCapture(i).isOpened()) for i in range(5)]"
```

---

## 📸 Screenshots

| View | Description |
|------|-------------|
| Upload Zone | Drag & drop area with file browser |
| Detection View | Original image with drawn bounding boxes and labels |
| Summary Panel | Severity level, class counts, affected area % |
| Detection Cards | Per-detection breakdown: confidence, severity, coordinates |
| Realtime HUD | Live FPS counter, threshold indicator, class counts overlay |

---

## 📦 Tech Stack

| Layer         | Technology                     |
|---------------|--------------------------------|
| Model         | YOLOv8s (Ultralytics)          |
| Training      | Google Colab (T4 GPU)          |
| Backend       | FastAPI 0.111, Python 3.11     |
| Inference     | Ultralytics + OpenCV           |
| Frontend      | React 18, Vite 5               |
| Styling       | Inline CSS (no deps)           |
| Logging       | Python logging + JSONL         |
| Deployment BE | Railway / Render               |
| Deployment FE | Vercel                         |

---

## 📄 License

MIT License — Free for academic and commercial use.

---

*Built with ❤️ — Road Surface Damage Detection System v1.0*
