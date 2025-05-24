#include "../include/TSDR/image_handler.h"
#include "../config/config.h"
#include "../include/TSDR/draw.h"
#include <opencv2/opencv.hpp>

ImageProcessor::ImageProcessor() : interface_engine() {}

void ImageProcessor::process(const std::string& image_path) {
    // Load the image from the given file path
    cv::Mat image = cv::imread(image_path);

    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return;     // Exit if image could not be loaded
    }

    std::vector<Detection> detections;

    // Run detection and recognition on the loaded image
    auto predictions = interface_engine.engine(image, detections);

    // Draw bounding boxes and CNN predictions on the image
    draw_detections_with_cnn_predictions(image, detections, predictions, CONF_THRESHOLD);

    // Display the annotated image in a window
    cv::imshow("Image Prediction", image);
    cv::waitKey(0);
}