# 🚦 TrafficLens — Real-Time Vehicle Detection with YOLOv8

TrafficLens ek vehicle detection system hai jo traffic footage me **car, bus, truck aur van** ko detect karta hai. Model ko **UA-DETRAC** dataset par fine-tune kiya gaya hai aur trained model ko real road video par test karke annotated output video generate hoti hai.

> **Result:** 96.7% mAP@0.5 aur 84.6% mAP@0.5:0.95 validation set par, ~4.7 ms per image inference speed (Tesla T4).

---

## Features

- 4 vehicle classes ki detection: `car`, `bus`, `truck`, `van`
- Pretrained YOLOv8s ko transfer learning se fine-tune kiya gaya
- Roboflow se dataset ka seamless download (YOLO format)
- Video par frame-by-frame inference (OpenCV) aur bounding boxes ke saath output video
- FFmpeg se output video compress karne ka pipeline (H.264)
- Poora workflow Google Colab (free T4 GPU) par chalta hai

---

## Tech Stack

| Area | Tool |
|---|---|
| Model | YOLOv8s (Ultralytics) |
| Deep Learning Framework | PyTorch (CUDA) |
| Dataset & Annotation | Roboflow, UA-DETRAC (10K version) |
| Video Processing | OpenCV |
| Video Compression | FFmpeg (libx264) |
| Augmentation | Mosaic, MixUp, rotation, translate, scale, shear + Albumentations |
| Optimizer | SGD (lr0 = 0.01, momentum = 0.937) |
| Environment | Google Colab, Tesla T4 GPU, Python 3.11 |

---

## Dataset

- **Source:** [UA-DETRAC Dataset 10K](https://universe.roboflow.com/) (Roboflow Universe, workspace `rjacaac1`, version 2)
- **Format:** YOLOv11-style annotations (YOLO txt format)
- **Split:** 9,316 training images / 500 validation images
- **Classes (4):** `bus`, `car`, `truck`, `van`
- **Validation instances:** 4,845 objects

---

## Training Configuration

| Parameter | Value |
|---|---|
| Base weights | `yolov8s.pt` (COCO pretrained) |
| Epochs | 20 |
| Image size | 640 × 640 |
| Batch size | 16 |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Mosaic / MixUp | Enabled |
| Rotation / Translate / Scale / Shear | 10° / 0.1 / 0.5 / 2 |
| Parameters | ~11.1M |
| GFLOPs | 28.4 |
| Training time | ~1.3 hours on Tesla T4 |

---

## Results

Validation set (500 images) par final performance:

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| **All** | 0.940 | 0.930 | **0.967** | **0.846** |
| Car | 0.961 | 0.942 | 0.987 | 0.837 |
| Bus | 0.966 | 0.917 | 0.944 | 0.813 |
| Truck | 0.892 | 0.872 | 0.947 | 0.829 |
| Van | 0.940 | 0.987 | 0.991 | 0.905 |

**Speed:** 0.5 ms preprocess · 4.7 ms inference · 3.2 ms postprocess (per image)

Training ke dauran mAP@0.5 epoch 1 ke 0.789 se badh kar epoch 20 tak 0.967 ho gaya, aur loss curves lagatar neeche aaye.

<!-- Apni images yahan add karo:
![Training Results](assets/results.png)
![Confusion Matrix](assets/confusion_matrix.png)
![Demo](assets/demo.gif)
-->

---

## Project Structure

```
TrafficLens/
├── notebooks/
│   ├── Yolov11_training.ipynb     # Dataset download + model training + validation
│   └── Yolov11_testing.ipynb      # Video inference + output compression
├── weights/
│   └── best.pt                    # Trained weights
├── assets/                        # Screenshots, result plots, demo gif
└── README.md
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install ultralytics roboflow opencv-python
# Optional (for FFmpeg compression)
sudo apt install ffmpeg
```

### 2. Download the dataset

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("rjacaac1").project("ua-detrac-dataset-10k")
dataset = project.version(2).download("yolov11")
```

### 3. Train

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(
    data="UA-DETRAC-DATASET-10K-2/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    optimizer="SGD",
    lr0=0.01,
    mosaic=True,
    mixup=True,
    degrees=10, translate=0.1, scale=0.5, shear=2,
)
```

### 4. Run inference on a video

```python
from ultralytics import YOLO
import cv2

model = YOLO("weights/best.pt")
cap = cv2.VideoCapture("traffic.mp4")

w, h = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    result = model.predict(frame, conf=0.3, imgsz=640, verbose=False)[0]
    out.write(result.plot())

cap.release()
out.release()
```

### 5. Compress the output (optional)

```bash
ffmpeg -i output.mp4 -vcodec libx264 -crf 28 output_compressed.mp4 -y
```

---

## Future Improvements

- Multi-object tracking (ByteTrack / BoT-SORT) ke saath vehicle counting
- Zyada epochs aur YOLOv11 / larger model variants ke saath comparison
- Speed estimation aur lane-wise traffic analysis
- Streamlit / Gradio web demo
- ONNX / TensorRT export for faster edge deployment

---

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Roboflow](https://roboflow.com/) for dataset hosting
- UA-DETRAC dataset authors

---

.
