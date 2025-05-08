#include "include/detection.h"
#include "include/yolo_utils.h"
#include <opencv2/opencv.hpp>

cv::dnn::Net loadYoloModel(const std::string& model_path) {
    cv::dnn::Net yolo_net = cv::dnn::readNetFromONNX(model_path);

    std::cout << "Model loaded successfully from: " << model_path << std::endl;
    std::vector<cv::String> input_names = yolo_net.getLayerNames();
    std::cout << "Model has " << input_names.size() << " layers." << std::endl;

    if (yolo_net.empty()) {
        throw std::runtime_error("Failed to load the YOLO model");
    }
    return yolo_net;
}

std::vector<Detection> post_process(const cv::Mat& output, const cv::Size& image_size) {
    std::vector<Detection> detections;
    std::cout << "Output shape: " << output.size << std::endl;
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

void draw_detections(const cv::Mat& image, const std::vector<Detection>& detections, float CONF_THRESHOLD) {
    for (const auto& det : detections) {
        if (det.confidence < CONF_THRESHOLD) {
            continue;
        }

        cv::Rect box = det.box & cv::Rect(0, 0, image.cols, image.rows);
        if (box.width <= 0 || box.height <= 0) {
            std::cerr << "Warning: Invalid box dimensions: " << box << std::endl;
            continue;
        }

        rectangle(image, box, cv::Scalar(0, 255, 0), 2);

        std::string label = "ID:" + std::to_string(det.class_id) + " (" + cv::format("%.2f", det.confidence) + ")";
        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int top = std::max(box.y - labelSize.height - baseLine, 0);
        int left = std::max(box.x, 0);
        cv::rectangle(image, cv::Point(left, top),
                      cv::Point(left + labelSize.width, top + labelSize.height + baseLine),
                      cv::Scalar(0, 255, 0), cv::FILLED);

        cv::putText(image, label, cv::Point(left, top + labelSize.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}