#ifndef YOLO_UTILS_H
#define YOLO_UTILS_H

#pragma once

extern const float CONF_THRESHOLD;
extern const float NMS_THRESHOLD;
extern const float IOU_THRESHOLD;
extern const int INPUT_WIDTH;
extern const int INPUT_HEIGHT;
// Performance Metrics
extern int total_TP;
extern int total_FP;
extern int total_FN;

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

// cv::dnn::Net loadYoloModel(const std::string& model_path);

std::vector<cv::Rect> load_lables(const std::string& label_path, const cv::Size& image_size);


// std::vector<Detection> post_process(const cv::Mat& output, const cv::Size& image_size);
// 
void draw_detections(const cv::Mat& image, const std::vector<Detection>& detections, float CONF_THRESHOLD);

float calculateIoU(const cv::Rect& pred_box, const cv::Rect& true_box);

void evaluateIoU(const std::vector<Detection>& detections, const std::vector<cv::Rect>& ground_truths, int& total_TP, int& total_FP, int& total_FN, float IOU_THRESHOLD);

std::vector<cv::Mat> run_yolo(const cv::Mat& img, cv::dnn::Net& yolo_net, std::string label_path);

void calculateFinalMetrics(int total_TP, int total_FP, int total_FN);

#endif
