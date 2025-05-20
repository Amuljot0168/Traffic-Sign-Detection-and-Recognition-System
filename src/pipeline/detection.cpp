// detection.cpp
// Contains logic for loading YOLO model, running YOLO, processing the results and cropping the detections for CNN input
#include "../include/TSDR/detection.h"
#include "../config/config.h"
#include <opencv2/opencv.hpp>

// Loads YOLO model
cv::dnn::Net load_yolo_model(const std::string& model_path) {
    cv::dnn::Net yolo_net = cv::dnn::readNetFromONNX(model_path);

    std::cout << "Model loaded successfully from: " << model_path << std::endl;
    std::vector<cv::String> input_names = yolo_net.getLayerNames();
    std::cout << "Model has " << input_names.size() << " layers." << std::endl;

    if (yolo_net.empty()) {
        throw std::runtime_error("Failed to load the YOLO model");
    }
    return yolo_net;
}

// Runs YOLO Model and returns raw YOLO output
std::vector<cv::Mat> run_yolo(const cv::Mat& frame, cv::dnn::Net& yolo_net) {
    const int input_width = 415;
    const int input_height = 415;

    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(input_width, input_height), cv::Scalar(), true, false);

    auto start = std::chrono::high_resolution_clock::now();
    yolo_net.setInput(blob);
    
    std::vector<cv::Mat> outputs;
    yolo_net.forward(outputs, yolo_net.getUnconnectedOutLayersNames());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "YOLO INFERENCE TIME: " << duration.count() << " MS" << std::endl;
    return outputs;
}

// 
std::vector<Detection> post_process_yolo(const cv::Mat& output, const cv::Size& image_size) {
    std::vector<Detection> detections;

    int rows = output.size[1];
    int dimensions = output.size[2];
    const float* data = (float*)output.data;
    
    for(int i = 0; i < rows; ++i) {
        float object_conf = data[i * dimensions + 4];
        if(object_conf < CONF_THRESHOLD)
            continue;
        
        float max_class_score = -1.0f;
        int class_id = -1;
        for(int j  = 5; j < dimensions; ++j) {
            float class_score = data[i * dimensions + j];
            if(class_score > max_class_score) {
                max_class_score = class_score;
                class_id = j - 5;
            }
        }
            
        float confidence = object_conf * max_class_score;
        if(confidence < CONF_THRESHOLD)
        continue;
            
        float cx = data[i * dimensions + 0];
        float cy = data[i * dimensions + 1];
        float  w = data[i * dimensions + 2];
        float  h = data[i * dimensions + 3];
        float x = cx - w / 2.0f;
        float y = cy - h / 2.0f;
        float scale_x = (float)image_size.width / 415.0f;
        float scale_y = (float)image_size.height / 415.0f;

        int left = int(x * scale_x);
        int top = int(y * scale_y);
        int width = int(w * scale_x);
        int height = int(h * scale_y);

        cv::Rect box(left, top, width, height);

        detections.push_back({ class_id, confidence, box });
    }
        
    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
        
    for(const auto& d: detections) {
        boxes.push_back(d.box);
        scores.push_back(d.confidence);
    }
        
    std::vector<Detection> final_detections;
    cv::dnn::NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD, indices);
    
    for(int idx : indices)
    final_detections.push_back(detections[idx]);
        
    return final_detections;
}

std::vector<cv::Mat> crop_detection_roi(const cv::Mat& frame, const std::vector<Detection>& detections, const cv::Size& targetSize) {
    std::vector<cv::Mat> cropped_rois;

    for(auto& det : detections) {
        cv::Rect box = det.box & cv::Rect(0, 0, frame.cols, frame.rows);
        if (box.width <= 0 || box.height <= 0) continue;

        cv::Mat roi = frame(box);

        int maxDim = std::max(roi.cols, roi.rows);
        cv::Mat square = cv::Mat::zeros(maxDim, maxDim, roi.type());
        roi.copyTo(square(cv::Rect((maxDim - roi.cols) / 2, (maxDim - roi.rows) / 2, roi.cols, roi.rows)));

        cv::Mat resized;
        cv::resize(square, resized, targetSize);

        cropped_rois.push_back(resized);
    }
    
    return cropped_rois;
}