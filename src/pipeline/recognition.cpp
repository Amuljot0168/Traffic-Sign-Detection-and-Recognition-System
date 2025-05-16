// recognition.cpp
// Contains lofic for loading and running CNN model and processing output to get CNN prediction

#include "recognition.h"
#include <opencv2/opencv.hpp>

cv::dnn::Net load_cnn_model(const std::string& model_path) {
    cv::dnn::Net cnn_net = cv::dnn::readNetFromONNX(model_path);

    std::cout << "Model loaded successfully from: " << model_path << std::endl;
    std::vector<cv::String> input_names = cnn_net.getLayerNames();
    std::cout << "Model has " << input_names.size() << " layers." << std::endl;

    if (cnn_net.empty()) {
        throw std::runtime_error("Failed to load the CNN model");
    }
    return cnn_net;
}

cv::Mat preprocess_images_for_cnn(cv::Mat& img) {
    cv::Mat resized, blob;
    cv::resize(img, resized, cv::Size(32, 32));

    blob = cv::dnn::blobFromImage(resized, 1.0 / 255.0, cv::Size(32, 32), cv::Scalar(), false, false);

    return blob;
}

cv::Mat run_cnn(cv::Mat& blob, cv::dnn::Net& cnn_net) {
    cnn_net.setInput(blob);
    cv::Mat output = cnn_net.forward();

    return output;
}

int cnn_inference(cv::Mat output) {
    cv::Point class_id_point;
    double confidence;
    minMaxLoc(output, 0, &confidence, 0, &class_id_point);
    int predicted = class_id_point.x;

    return predicted;
}