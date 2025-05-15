// config template
#pragma once
#include <string>

// Paths to model
const std::string YOLO_MODEL_PATH = "models/yolov5s_detection.onnx";
const std::string CNN_MODEL_PATH = "models/cnn_recognition.onnx";

// Tunable hyperparameters
const float CONF_THRESHOLD = 0.40;
const float NMS_THRESHOLD = 0.45;
