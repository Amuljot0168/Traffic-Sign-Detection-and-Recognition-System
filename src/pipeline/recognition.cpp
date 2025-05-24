// recognition.cpp
// Contains lofic for loading and running CNN model and processing output to get CNN prediction

#include "../include/TSDR/recognition.h"
#include <opencv2/opencv.hpp>
#include <iostream>

Classifier::Classifier(const std::string& model_path, int input_dim) : input_size(input_dim) {
    cnn_net = cv::dnn::readNetFromONNX(model_path);

    if(cnn_net.empty()) {
        throw std::runtime_error("Failed to load the CNN model from " + model_path);
    }
    // std::cout << "Model loaded successfully from: " << model_path << std::endl;
}

// Preprocesses the input image: resizes and converts it into a blob for the CNN
cv::Mat Classifier::preprocess(const cv::Mat& img) {
    cv::Mat resized, blob;
    cv::resize(img, resized, cv::Size(input_size, input_size));
    blob = cv::dnn::blobFromImage(resized, 1.0 / 255.0, cv::Size(input_size, input_size), cv::Scalar(), false, false);
    return blob;
}

// Performs inference using the CNN on the given blob
cv::Mat Classifier::infer(const cv::Mat& blob) {
    cnn_net.setInput(blob);
    return cnn_net.forward();
}

// Predicts the class index for a given image using the CNN model
int Classifier::predict(const cv::Mat& img) {
    cv::Mat blob = preprocess(img);
    cv::Mat output = infer(blob);

    // Get the class index with the highest confidence score
    cv::Point class_id_point;
    double confidence;
    minMaxLoc(output, nullptr, &confidence, nullptr, &class_id_point);
    return class_id_point.x;  // predicted class index
}