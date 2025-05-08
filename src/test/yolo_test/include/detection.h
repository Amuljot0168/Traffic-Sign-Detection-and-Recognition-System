#ifndef DETECTION_H
#define DETECTION_H

#include <opencv2/opencv.hpp>
#include "yolo_utils.h"

cv::dnn::Net loadYoloModel(const std::string& model_path);
std::vector<Detection> post_process(const cv::Mat& output, const cv::Size& image_size);
void draw_detections(const cv::Mat& image, const std::vector<Detection>& detections, float CONF_THRESHOLD);

#endif