
# Underwater Shipwreck Detection with YOLOv11

## Project Overview

This project implements shipwreck detection based on side-scan sonar images using the Ultralytics YOLOv11 framework. It provides training, validation, and inference pipelines for developing an efficient and lightweight underwater target detection model.

---

## Environment Setup

Install required dependencies:

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for efficient training.

---

## Dataset

We use the [Underwater Shipwreck Dataset](https://gitee.com/nwpu-r/underwater-shipwreck-dataset), a side-scan sonar image dataset designed for shipwreck detection tasks.

**Dataset Structure Example:**

```
datasets/shipwreck/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── shipwreck.yaml
```




## Scripts Overview

### 1. Training (`train.py`)

Train a YOLOv11 model on the shipwreck dataset with configurable hyperparameters.

**Key parameters:**

- `data`: Path to dataset `.yaml` file
- `epochs`: Number of training epochs
- `imgsz`: Image size
- `batch`: Batch size
- `device`: GPU ID
- `resume`: Resume training from last checkpoint

**Example:**

```bash
python train.py
```

---

### 2. Validation (`val.py`)

Evaluate a trained model on the validation set and compute detection metrics such as mAP, Precision, and Recall.

**Example:**

```bash
python val.py --weights runs/train/exp/weights/best.pt --data ./datasets/shipwreck/shipwreck.yaml --imgsz 640 --device 0
```

---

### 3. Inference (`predict.py`)

Run inference on images, videos, or live streams using a trained model.

**Key options:**

- `weights`: Path to trained model
- `source`: Path to images, video, or webcam stream

**Example:**

```bash
python predict.py --weights runs/train/exp/weights/best.pt --source ./datasets/shipwreck/images/test  
```

---

## Project Structure

```
.
├── datasets/
│   └── shipwreck/
│       ├── images/
│       ├── labels/
│       └── shipwreck.yaml
├── train.py
├── val.py
├── predict.py
├── yolo11l.pt
└── README.md
```

---

## Notes

- Training results, weights, and logs are saved under `runs/train/`.
- Validation produces a detailed performance report.
- Inference supports both batch image prediction and video stream detection.

---

## References

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [Underwater Shipwreck Dataset](https://gitee.com/nwpu-r/underwater-shipwreck-dataset)
