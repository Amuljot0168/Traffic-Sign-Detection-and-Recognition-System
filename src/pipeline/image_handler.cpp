#include "../include/TSDR/image_handler.h"
#include "../config/config.h"
#include "../include/TSDR/draw.h"
#include <opencv2/opencv.hpp>

ImageProcessor::ImageProcessor() : interface_engine() {}

void ImageProcessor::process(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            return;
        }

        std::vector<Detection> detections;
        auto predictions = interface_engine.engine(image, detections);

        draw_detections_with_cnn_predictions(image, detections, predictions, CONF_THRESHOLD);

        cv::imshow("Image Prediction", image);
        cv::waitKey(0);
}