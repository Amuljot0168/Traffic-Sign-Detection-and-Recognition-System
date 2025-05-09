#ifndef RECOGNITION_H
#define RECOGNITION_H
#include <opencv2/opencv.hpp>

cv::dnn::Net load_cnn_model(const std::string& model_path);
cv::Mat preprocess_images_for_cnn(cv::Mat& img);
cv::Mat run_cnn(cv::Mat& img, cv::dnn::Net& cnn_net);
int cnn_inference(cv::Mat output);


#endif