#ifndef DRAW_H
#define DRAW_H
#include "../include/TSDR/detection.h"
#include <opencv2/opencv.hpp>

void draw_detections(const cv::Mat& image, const std::vector<Detection>& detections, float CONF_THRESHOLD);
void draw_detections_with_cnn_predictions(cv::Mat& image, const std::vector<Detection>& detections, const std::vector<std::string>& cnn_predictions, float CONF_THRESHOLD);
#endif