#ifndef DETECTION_H
#define DETECTION_H
#include <opencv2/opencv.hpp> 
struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

cv::dnn::Net load_yolo_model(const std::string& model_path);
std::vector<cv::Mat> run_yolo(const cv::Mat& img, cv::dnn::Net& yolo_net);
std::vector<Detection> post_process_yolo(const cv::Mat& output, const cv::Size& image_size);
std::vector<cv::Mat> crop_detection_roi(const cv::Mat& frame, const std::vector<Detection>& detections, const cv::Size& targetSize = cv::Size(32, 32));
#endif