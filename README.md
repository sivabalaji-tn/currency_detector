# 🇮🇳 Indian Currency Detection System

### Deep Learning-Based Real-Time ₹50 Currency Detection Using YOLOv8

---

## <span style="color:red">⚠️ IMPORTANT: DATASET DOWNLOAD REQUIRED</span>

# 🔴 THE DATASET IS NOT HOSTED ON GITHUB!

The dataset contains large image and annotation files that exceed GitHub's practical file hosting limits.

**Please visit the Google Drive link below to download the complete dataset.**

### 📂 [👉 CLICK HERE TO DOWNLOAD THE DATASET FROM GOOGLE DRIVE](https://drive.google.com/file/d/1YJJGg2CEcK-mXt8Yo9a9cPLnLEaKDaFW/view?usp=sharing)

**After downloading, extract the dataset into the project root directory.**

The folder structure should look like this:

```text
currency-detector/
│
├── dataset/
│   ├── collection/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── data.yaml
│   └── classes.txt
│
├── models/
│   └── onnx/
│       └── model.onnx
│
├── test.py
└── README.md
```

---

## 🚀 Running the Project

### 1. Install dependencies

```bash
pip install opencv-python numpy
```

### 2. Run the currency detector

```bash
python test.py --model models/onnx/model.onnx
```

### 3. Webcam Detection

Allow webcam access and show an Indian ₹50 note in front of the camera.

---

## 🛠️ Technology Stack

- Python
- YOLOv8n
- Deep Learning
- Computer Vision
- OpenCV
- NumPy
- ONNX
- OpenCV DNN

## 📌 Project Overview

This project is designed to detect Indian ₹50 currency notes in real time using a webcam. A trained YOLOv8 object detection model is exported to ONNX format and executed using OpenCV DNN.

**Note:** The current model detects ₹50 notes but has not yet been verified to reliably distinguish newer ₹50 notes from older ₹50 notes.

---

**Developed as an academic project.**
