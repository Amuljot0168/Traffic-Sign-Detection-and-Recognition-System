# Architecture – Traffic Sign Detection and Recognition System

This document provides a comprehensive explanation of the architecture and components that power the **Traffic Sign Detection and Recognition System**, a real-time application for identifying Indian traffic signs using deep learning models integrated into a C++ pipeline.

---

## System Pipeline Overview

The application follows a sequential yet modular processing pipeline:

## Block Diagram

```
┌───────────────────────┐
│  Input (Image/Video)  │
└────────────┬──────────┘
             │
             ▼
┌───────────────────────┐
│ YOLOv5 Detection      │  ← Trained in Python, exported to ONNX
│ (ONNX model)          │
└────────────┬──────────┘
             │
             ▼
┌───────────────────────┐
│ Crop Detected Signs   │
│ (Bounding Boxes)      │
└────────────┬──────────┘
             │
             ▼
┌──────────────────────────────┐
│ CNN Classifier               │ ← Trained in Python, exported to ONNX/XML
│ (OpenVINO or ONNX model)     │
└────────────┬─────────────────┘
             │
             ▼
┌────────────────────────┐
│ Final Classified Label │
└────────────────────────┘
```

Each stage of the pipeline performs a well-defined task and communicates with the next through clean interfaces, ensuring maintainability and extensibility.

- **Input**

  - Accepts either an image (`--image path.jpg`) or video (`--video path.mp4`).
  - Loaded using OpenCV in C++.

- **YOLOv5 Detection**

  - Detects all traffic signs in the frame.
  - Trained using Python and Ultralytics YOLOv5.
  - Exported to ONNX for C++ inference via OpenCV DNN.

- **Crop Detected Signs**

  - For each bounding box from YOLOv5, the region is cropped.
  - Cropped ROIs are resized/preprocessed to match CNN input shape.

- **CNN Classifier**

  - Classifies the cropped ROI into a traffic sign class (e.g., stop, speed limit).
  - Trained in Python with TensorFlow/Keras.
  - Exported to:
    - ONNX: Generic, portable format.
    - XML (IR format): For optimized OpenVINO inference.

- **Final Label Output**
  - Displays the class name on the original image/video.
  - Bounding boxes and labels are rendered using OpenCV drawing tools.

---

## Component Breakdown & Responsibilities

### 1. **Input (Image or Video)**

- The application supports both static images and video files as input.
- Input is provided via command-line arguments (`--image` or `--video`).
- Frames are captured using **OpenCV** and passed into the detection module.

### 2. **YOLOv5 – Traffic Sign Detection**

- **Purpose**: Detects traffic signs from the input frame by outputting bounding boxes.
- **Framework**: Trained in **Python** using the **Ultralytics YOLOv5** library.
- **Export**: The trained `.pt` model is exported to ONNX format using:

  ```python
  model = YOLO("yolov5n.pt")
  model.export(format="onnx", dynamic=True)
  ```

- **Inference**: The ONNX model is loaded and run in C++ using the OpenCV DNN module (`cv::dnn::readNetFromONNX`).
- **Output**: For each detection, the model returns class ID, confidence, and bounding box coordinates.

### 3. **Region of Interest (ROI) Extraction**

- Each bounding box is used to **crop** the relevant portion of the frame.
- These cropped images are normalized/resized to match the CNN classifier input format.
- Optional preprocessing steps (resizing, mean subtraction, normalization) are done using OpenCV.

### 4. **CNN Classifier – Traffic Sign Recognition**

- **Purpose**: Performs **fine-grained classification** of the cropped traffic signs.
- **Architecture**: Typically a small CNN trained to recognize traffic sign classes (e.g., stop, speed limit, yield).
- **Frameworks**:
  - Can be trained using **TensorFlow**, **Keras**, or **PyTorch**.
  - Exported to:
    - **ONNX** for general compatibility.
    - Or **OpenVINO IR format** (`.xml`, `.bin`) for performance-optimized inference.
- **Inference in C++**:
  - Loaded via OpenCV DNN using:
    - `cv::dnn::readNetFromONNX()` for ONNX models
    - `cv::dnn::readNet()` for OpenVINO IR models

### 5. **Final Label Output**

- The CNN's output (softmax probability vector) is interpreted to assign a final label (class name).
- This label is rendered onto the original image frame using OpenCV drawing utilities.
- Optional: bounding box and label are color-coded for better visualization.

---

## Cross-language Data Flow: Python → C++

| Stage                  | Language | Action/Details                                                     |
| ---------------------- | -------- | ------------------------------------------------------------------ |
| Model Training         | Python   | YOLOv5 and CNN models are trained with datasets (e.g., GTSDB)      |
| Model Export           | Python   | Models are exported to ONNX / XML using Ultralytics / OpenVINO     |
| Model Inference Engine | C++      | Only inference logic is implemented in C++ using OpenCV            |
| Communication Bridge   | Files    | Models are shared between Python & C++ via `.onnx` or `.xml` files |

- **No runtime communication** is needed between Python and C++.
- Models are treated as plug-and-play assets in the C++ application.

---

## Inference Workflow (C++ Runtime)

1. Load YOLOv5 model (`yolov5n.onnx`)
2. Load CNN classifier (`traffic_cnn_model.xml`)
3. For each input frame:
   - Detect all traffic signs via YOLO
   - Crop each bounding box
   - Run CNN on each crop
   - Label and display results

This workflow ensures real-time performance and efficient memory handling using OpenCV’s optimized DNN module.

---

## Summary Table

| Component      | Role                               | Technology Used                     |
| -------------- | ---------------------------------- | ----------------------------------- |
| YOLOv5         | Detect traffic signs               | Python (Ultralytics YOLO), ONNX     |
| CNN Classifier | Classify type of traffic sign      | TensorFlow/Keras → OpenVINO or ONNX |
| OpenCV (DNN)   | Model Inference & Image Processing | C++                                 |
| User Interface | Input/output via CLI & OpenCV GUI  | C++                                 |
| Data Flow      | Export from Python, Load in C++    | Shared model files                  |

---

## Advantages of This Architecture

- **Modularity**: Easy to replace models without touching C++ code.
- **Cross-platform**: C++ backend runs on Windows/Linux; models are platform-agnostic.
- **Performance**: ONNX + OpenCV = GPU/CPU acceleration without heavy frameworks.
- **Scalability**: Can extend with new classifiers, webcam input, or GUI modules.

---
