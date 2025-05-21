#ifndef RECOGNITION_H
#define RECOGNITION_H

#include <opencv2/opencv.hpp>
#include <string>

class Classifier {
private: 
    cv::dnn::Net cnn_net;
    int input_size;

public:
    // Constructor to load the model
    Classifier(const std::string& model_path, int input_dim = 32);

    // Preprocess image
    cv::Mat preprocess(const cv::Mat& img);

    // Run prediction on a single frame
    int predict(const cv::Mat& img);

    // Run CNN forward pass on blob
    cv::Mat infer(const cv::Mat& blob);

};

#endif