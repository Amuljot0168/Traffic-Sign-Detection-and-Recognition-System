#ifndef DETECTION_H
#define DETECTION_H

#include <opencv2/opencv.hpp> 
#include <vector>
#include <string>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

class Detector {
private: 
    cv::dnn::Net yolo_net;
    float conf_threshold;
    float nms_threshold;
    cv::Size target_crop_size;
    const int input_width = 416;
    const int input_height = 416;

public:
    // Constructor to load the model, set thresholds and crop size
    Detector(const std::string& model_path, float conf_threshold, float nms_threshold, cv::Size crop_size = cv::Size(32, 32));

    // Run detection pipeline on an input image and return detections
    std::vector<Detection> detect(const cv::Mat& frame);

    // Crop ROIs for CNN input
    std::vector<cv::Mat> crop_rois(const cv::Mat& frame, const std::vector<Detection>& detections);
};

#endif