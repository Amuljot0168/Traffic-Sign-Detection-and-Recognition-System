# Inference Pipeline (C++ with OpenCV DNN)

## Objective
Load the trained ONNX model (YOLOv5 or CNN classifier) using OpenCV's DNN module in C++, perform inference on images or real-time video streams, and output annotated results efficiently.


---

## Key Components & Call Flow

```
Input Frame (Video/Image)
        │
        ▼
    YOLO Detection
        │
        ▼
Filter by Confidence & Allowed Classes
        │
        ▼
  Apply NMS (Non-Max Suppression)
        │
        ▼
 Crop Detected ROIs 
        │
        ▼
    CNN Classification
        │
        ▼
  Output: Class Labels + Boxes (filtered)

```

## Load ONNX Model

Loads the ONNX model

Sets backend and target (CPU/GPU)

##  Preprocessing Input Frame

Normalizes pixel values (e.g., divide by 255)

Resizes to model input size (e.g., 416×416 for YOLO)

Converts image to blob format

##  Parsing Detection Output 
* The YOLO model outputs tensors encoding bounding boxes, class scores, and class IDs.

* You extract these elements and filter detections by a confidence threshold to discard weak predictions.

* Apply Non-Maximum Suppression (NMS) to remove overlapping boxes and keep only the most confident detection per object.

* This step reduces false positives and cluttered overlapping boxes.
```
+---------------------+
|   Output Tensor(s)   |
+----------+----------+
           |
           v
+---------------------+
| Extract Boxes,      |
| Scores, Class IDs   |
+----------+----------+
           |
           v
+---------------------+
| Confidence Threshold |
+----------+----------+
           |
           v
+---------------------+
| Non-Max Suppression  |
+---------------------+
```

## Classification Output Parsing
* For each cropped detected region (ROI), pass it through the CNN classifier.

* The classifier outputs raw scores (logits) which are converted to probabilities using a softmax layer.

* Select the class with the highest probability.

* Map this class index to a human-readable label.


```
+----------------+
| Softmax Output |
+-------+--------+
        |
        v
+----------------+
| Find Max Class  |
+-------+--------+
        |
        v
+----------------+
| Map to Label   |
+----------------+
```
## Annotating & Displaying Results
* Once detections and classifications are finalized, annotate the original frame:

    Draw bounding boxes around detected objects.

     Put class labels (and optionally confidence scores) near the boxes.

* Display the annotated frame on-screen or save it to disk.

* Use OpenCV’s drawing functions for visualization.
```
+-------------------------+
|    Annotate Image       |
|  - Draw Boxes           |
|  - Put Class Labels     |
+------------+------------+
             |
             v
+-------------------------+
|   Show or Save Output   |
+-------------------------+
```

## Handling Real-Time Video Stream
* Capture video .

* In a loop, read frames continuously.

* For each frame:

   Preprocess and run the detection + classification pipeline.

    Annotate and display the frame.

* Allow exit on user input (e.g., pressing a key).

* Efficient resource management here ensures smooth, real-time performance.


```
+---------------------+
| Open Video Capture   |
+----------+----------+
           |
           v
+---------------------+
|   Read Frame Loop   |
+----------+----------+
           |
           v
+---------------------+
|    Run Inference    |
+----------+----------+
           |
           v
+---------------------+
|    Show Annotated   |
|      Frame          |
+----------+----------+
           |
           v
+---------------------+
| Exit on Key Press   |
+---------------------+
```

