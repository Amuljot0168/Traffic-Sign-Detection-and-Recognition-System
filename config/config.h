#pragma once
#include <string>

// Paths to models
// const std::string YOLO_MODEL_PATH = "D:/KPIT/opencvtest/models/v5-4-class.onnx";
const std::string YOLO_MODEL_PATH = "D:/KPIT/opencvtest/models/yolov5n_detection.onnx";
const std::string CNN_MODEL_PATH  = "D:/KPIT/opencvtest/models/cnn_recognition.onnx";

// Paths to test data
// const std::string IMAGE_PATH = "D:/KPIT/opencvtest/test_data/YOLO_Test/images/00882.jpg";
const std::string VIDEO_PATH = "D:/KPIT/opencvtest/test_data/video_input/VID-20250508-WA0003.mp4";

// Tunable hyperparameters
const float CONF_THRESHOLD = 0.40;
const float NMS_THRESHOLD = 0.45;

