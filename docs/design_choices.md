<!-- Why YOLOv5 (vs SSD, Faster R-CNN)

Why CNN (vs using YOLOv5 for classification too)

Why ONNX (interoperability between Python training and C++ deployment)

Why OpenCV for C++ inference

 -->
 # Design Choices  
This document outlines the key architectural and model selection decisions made during the development of the project. Each choice was driven by a balance between accuracy, speed, modularity, and future extensibility.  

## Architectural Design  
### 1. Separation of Detection and Classification  
Instead of relying on a monolithic model for object detection and classification, the system uses a modular pipeline:

- YOLOv5 for object detection
- A custom CNN classifier for post-detection classification

#### Rationale:
- Separation allows better control over each stage of inference.
- Easier to swap models independently for benchmarking or upgrades.
- Allows fine-tuning each component separately (e.g., using a lightweight YOLO variant with a heavier classifier if needed).

### 2. Use of InterfaceEngine as a Central Coordinator

The ```InterfaceEngine``` class was introduced to encapsulate the interaction between detection and classification stages.

#### Rationale:
- Keeps the processing logic centralized and abstracted.
- Prevents redundancy between ImageProcessor and VideoProcessor.
- Makes the system more extensible (e.g., adding new input types like RTSP streams or webcam feeds becomes easier).

### 3. Minimal Dependencies in Core Classes
Each class was written with minimal direct dependencies and encapsulated logic.

#### Rationale:
- Improves testability (you can test Classifier or Detector independently).
- Reduces coupling, making future refactoring or model switching simpler.

### 4. Performance Tracking and Frame Skipping
The VideoProcessor includes FPS tracking and supports frame-skipping to optimize real-time inference.

#### Rationale:
- Balances CPU/GPU load on lower-end hardware.
- Ensures a smoother visual output when deployed on real-time video feeds.

##  Model Choices

### 1. YOLOv5n and YOLOv5s
Both YOLOv5n (nano) and YOLOv5s (small) variants are included.
- YOLOv5n offers faster inference but lower accuracy.
- YOLOv5s provides significantly more accurate detections at the cost of speed.

#### Rationale:
- Offers users a configurable speed vs. accuracy tradeoff.
- Both models are small enough (~27MB) for efficient C++ deployment.

**Note**: Users can train their own YOLO model using the provided Python script and convert it to ONNX. For YOLOv8 or newer models, using ONNX Runtime is recommended.

### 2. Custom CNN Classifier
The classifier is a lightweight custom CNN trained on a smaller, curated dataset of cropped detection outputs.

#### Rationale:
- Keeps classification fast and efficient during inference.
- Offers high accuracy (95.04%) on the test dataset.
- Avoids the overhead of using larger pre-trained networks like ResNet or MobileNet for a simple multi-class task.


##  Configurability
The system supports key tunable parameters via a config file:

- Model path (choose between YOLOv5n or YOLOv5s)
- Confidence threshold
- NMS threshold
- Frame skip rate

#### Rationale:
- Offers flexibility without code changes.
- Makes it easy to deploy the system across various environments and use cases.


## Extensibility
The current setup is ready for enhancements such as:

- Switching to ONNX Runtime for newer models.
- Replacing CNN with a more complex classifier.
- Supporting different media types (webcam, live stream, etc.)
- Plug-and-play model updates without architecture changes.