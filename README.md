# 🚧 Pothole Detection & Road Damage Analysis

YOLOv8-powered road damage detection system with a Flask web interface.

---

## Project Files

| File | Description |
|------|-------------|
| `Pothole_Detection_Colab.ipynb` | Full training pipeline (Google Colab) |
| `app.py` | Flask web application |
| `requirements.txt` | Python dependencies |
| `pothole_best.pt` | ← Place your trained model here |

---

## Quick Start

### 1. Train the Model (Google Colab)
1. Open `Pothole_Detection_Colab.ipynb` in Google Colab
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Add your Roboflow API key in Step 2
4. Run all cells — model trains in ~15 min on T4
5. Download `pothole_best.pt` at the end

### 2. Run the Flask App
```bash
# Install dependencies
pip install -r requirements.txt

# Place your model in the project folder
cp ~/Downloads/pothole_best.pt .

# Start the server
python app.py

# Open browser
open http://localhost:5000
```

### 3. Without a trained model (Demo Mode)
The app runs in **Demo Mode** if no `.pt` file is found — it shows mock detections so the UI can be previewed immediately.

---

## Features

- **Image Upload** — Drag & drop JPG/PNG for instant pothole detection
- **Video Analysis** — Upload MP4/AVI road footage
- **Live Webcam** — Real-time MJPEG stream via `/webcam`
- **Severity Scoring** — High / Medium / Low based on pothole area
- **JSON Export** — Download full detection report
- **Confidence Controls** — Adjustable conf & IoU thresholds in UI

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/detect` | POST | Run detection on uploaded image (returns JSON + base64 image) |
| `/webcam` | GET | MJPEG webcam stream |
| `/health` | GET | Model status & config |

### `/detect` Request
```
POST /detect
Content-Type: multipart/form-data

file  = <image file>
conf  = 0.25   (optional)
iou   = 0.45   (optional)
```

### `/detect` Response
```json
{
  "total": 2,
  "inference_time_s": 0.041,
  "summary": { "high": 1, "medium": 1, "low": 0 },
  "image_b64": "...",
  "detections": [
    {
      "class": "pothole",
      "confidence": 0.87,
      "bbox": [120, 200, 310, 380],
      "area_pct": 3.4,
      "severity": "Medium"
    }
  ]
}
```

---

## Roboflow Datasets

Recommended public datasets:
- **Pothole Detection** — `roboflow-100/pothole-detection-system`
- **Road Damage** — search "road damage" on [roboflow.com/universe](https://universe.roboflow.com)

---

## Model Training Tips

| Setting | Recommended |
|---------|------------|
| Model | `yolov8s` (balanced speed/accuracy) |
| Epochs | 50–100 |
| Image size | 640 |
| Batch size | 16 (T4 GPU) |
| Augmentation | mosaic=1, fliplr=0.5 |
