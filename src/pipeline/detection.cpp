// detection.cpp
// Contains logic for loading YOLO model, running YOLO, processing the results and cropping the detections for CNN input
#include "../include/TSDR/detection.h"
#include "../config/config.h"
// #include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

Detector::Detector(const std::string& model_path, float conf_thresh, float nms_thresh, cv::Size crop_size) : conf_threshold(conf_thresh), nms_threshold(nms_thresh), target_crop_size(crop_size) {
    yolo_net = cv::dnn::readNetFromONNX(model_path);
    if(yolo_net.empty()) {
        throw std::runtime_error("Failed to load YOLO model from " + model_path);
    }
}

std::vector<Detection> Detector::detect(const cv::Mat& frame) {
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(input_width, input_height), cv::Scalar(), true, false);

    yolo_net.setInput(blob);
    
    std::vector<cv::Mat> outputs;
    yolo_net.forward(outputs, yolo_net.getUnconnectedOutLayersNames());

    // Post processing logic
    const cv::Mat& output = outputs[0];

    std::vector<Detection> detections;

    int rows = output.size[1];
    int dimensions = output.size[2];
    const float* data = (float*)output.data;

    for(int i = 0; i < rows; ++i) {
        float object_conf = data[i * dimensions + 4];
        if(object_conf < conf_threshold)
            continue;

        float max_class_score = -1.0f;
        int class_id = -1;
        for(int j = 5; j < dimensions; ++j) {
            float class_score = data[i * dimensions + j];
            if(class_score > max_class_score) {
                max_class_score = class_score;
                class_id = j - 5;
            }
        }

        float confidence = object_conf * max_class_score;
        if(confidence < conf_threshold)
            continue;

        float cx = data[i * dimensions + 0];
        float cy = data[i * dimensions + 1];
        float w = data[i * dimensions + 2];
        float h = data[i * dimensions + 3];
        float x = cx - w / 2.0f;
        float y = cy - h / 2.0f;

        float scale_x = (float)frame.cols / input_width;
        float scale_y = (float)frame.rows / input_height;

        int left = int(x * scale_x);
        int top = int(y * scale_y);
        int width = int(w * scale_x);
        int height = int(h * scale_y);

        cv::Rect box(left, top, width, height);

        detections.push_back({ class_id, confidence, box });
    }

    // Perform Non-Maximum Suppression (NMS) to remove overlapping boxes
    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    // Extract bounding boxes and their confidence scores from detections
    for(const auto& d: detections) {
        boxes.push_back(d.box);
        scores.push_back(d.confidence);
    }

    // Apply NMS to filter out overlapping boxes based on confidence and NMS threshold
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, indices);

    // Collect final detections after NMS filtering
    std::vector<Detection> final_detections;
    for(int idx : indices)
        final_detections.push_back(detections[idx]);

    return final_detections;
}

std::vector<cv::Mat> Detector::crop_rois(const cv::Mat& frame, const std::vector<Detection>& detections) {
    std::vector<cv::Mat> cropped_rois;

    for(auto& det : detections) {
        // Ensure the detection box is within image bounds
        cv::Rect box = det.box & cv::Rect(0, 0, frame.cols, frame.rows);
        if (box.width <= 0 || box.height <= 0) continue;

        // Crop the region of interest (ROI) from the frame
        cv::Mat roi = frame(box);

        // Create a square canvas with size equal to max dimension of ROI
        int maxDim = std::max(roi.cols, roi.rows);
        cv::Mat square = cv::Mat::zeros(maxDim, maxDim, roi.type());

        // Center the ROI on the square canvas
        roi.copyTo(square(cv::Rect((maxDim - roi.cols) / 2, (maxDim - roi.rows) / 2, roi.cols, roi.rows)));

        // Resize the squared ROI to the target input size for the CNN
        cv::Mat resized;
        cv::resize(square, resized, target_crop_size);

        // Add the processed ROI to the output vector
        cropped_rois.push_back(resized);
    }

    return cropped_rois;
}