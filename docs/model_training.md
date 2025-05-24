# Traffic Sign Detection and Recognition - Model Training

This document provides the **complete training setup** for both the **YOLOv5 object detection model** and a **CNN-based image classification model** 

---

## Dataset

### YOLOv5 Dataset
- **Location**: `/kaggle/input/traffic-signs-dataset-in-yolo-format/`
- **Format**: YOLO-format with images and corresponding label `.txt` files

### CNN Dataset
- **Location**: `/kaggle/input/gtsrb-german-traffic-sign/`
- **Structure**:
  - `Train.csv` (contains image paths and class labels)
  - `Train/` folder containing traffic sign images

---

## YOLOv5 Model Training

### Configuration
- **Config file**: `yolov3_ts_train.cfg`
- **Model**: YOLOv5s
- **Image size**: 416X416
- **Batch size**: 16
- **Epochs**: 100

### Training Procedure
A training script is executed, specifying the dataset path, pre-trained weights (e.g., yolov5s.pt), and other hyperparameters.

Training progresses through epochs, optimizing object classification and bounding box regression.

### Training Command
```bash
!wandb disabled
!python train.py 
--img 415 
--batch 16 
--data /kaggle/working/dataset/dataset.yaml 
--weights yolov5s.pt 
--cache 
--workers 2
```

### Output
A set of learned weights stored in a best.pt file, representing the model with highest validation performance.

These weights are later used for inference on unseen test images.

## YOLOv5s Architecture 
```
Input Image (416x416x3)
│
┌────▼────┐
│ Backbone│ ← CSPDarknet
└────┬────┘
│
┌────▼────┐
│ Neck │ ← PANet
└────┬────┘
│
┌────▼────┐
│ Head │ ← Detection
└─────────┘
│
┌────▼────┐
│ Output │
```


## CNN Classification Model Training

### Objective
To build a **Convolutional Neural Network (CNN)** that classifies cropped traffic sign images into one of the 43 predefined classes.

### Data Preparation
**Images** are resized to **32×32** pixels to ensure consistent input shape.

**Normalization** is performed to scale pixel values between 0 and 1.

The dataset is split into training and test subsets to evaluate model generalization.

###  CNN Architecture

The model uses __3 convolutional layers__ with increasing filters (32 → 64 → 128) and max pooling for spatial downsampling.

A fully connected dense layer maps the extracted features to the 43 output classes via a softmax layer.

Activation functions like ReLU introduce non-linearity, aiding learning of complex patterns.


### Training Process
It is evaluated on a hold-out test set to report classification accuracy.
```bash

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")
     
```

### Save the CNN Model

A trained CNN model capable of recognizing traffic sign types with high accuracy.

The final model is saved for future inference or export.

## CNN Architecture

```
Input Image (32x32x3)
        │
   ┌────▼────┐
   │ Conv2D  │ → 32 filters
   └────┬────┘
        │
   ┌────▼────┐
   │ MaxPool │
   └────┬────┘
        │
   ┌────▼────┐
   │ Conv2D  │ → 64 filters
   └────┬────┘
        │
   ┌────▼────┐
   │ MaxPool │
   └────┬────┘
        │
   ┌────▼────┐
   │ Conv2D  │ → 128 filters
   └────┬────┘
        │
   ┌────▼────┐
   │ Flatten │
   └────┬────┘
        │
   ┌────▼────┐
   │ Dense   │ → 128 units (ReLU)
   └────┬────┘
        │
   ┌────▼────┐
   │ Output  │ → 43 units (Softmax)
   └─────────┘

```

This document summarizes the training pipeline for both detection and classification of traffic signs using YOLOv5 and CNNs.
