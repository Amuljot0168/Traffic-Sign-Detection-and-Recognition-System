#include "../include/TSDR/draw.h"
#include "../include/TSDR/detection.h"
#include <opencv2/opencv.hpp>

void draw_detections(const cv::Mat& image, const std::vector<Detection>& detections, float CONF_THRESHOLD) {
    for (const auto& det : detections) {
        if (det.confidence < CONF_THRESHOLD) {
            continue;
        }

        cv::Rect box = det.box & cv::Rect(0, 0, image.cols, image.rows);
        if (box.width <= 0 || box.height <= 0) {
            std::cerr << "Warning: Invalid box dimensions: " << box << std::endl;
            continue;
        }

        rectangle(image, box, cv::Scalar(0, 255, 0), 2);

        std::string label = "Conf (" + cv::format("%.2f", det.confidence) + ")";
        // std::string label = "ID:" + std::to_string(det.class_id) + " (" + cv::format("%.2f", det.confidence) + ")";
        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int top = std::max(box.y - labelSize.height - baseLine, 0);
        int left = std::max(box.x, 0);
        cv::rectangle(image, cv::Point(left, top),
                      cv::Point(left + labelSize.width, top + labelSize.height + baseLine),
                      cv::Scalar(0, 255, 0), cv::FILLED);

        cv::putText(image, label, cv::Point(left, top + labelSize.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

void draw_detections_with_cnn_predictions(cv::Mat& image, const std::vector<Detection>& detections, const std::vector<std::string>& cnn_predictions, float CONF_THRESHOLD) {
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        if (det.confidence < CONF_THRESHOLD) {
            continue;
        }

        cv::Rect box = det.box & cv::Rect(0, 0, image.cols, image.rows);
        if (box.width <= 0 || box.height <= 0) {
            std::cerr << "Warning: Invalid box dimensions: " << box << std::endl;
            continue;
        }

        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);

    // Use CNN prediction if available
        std::string label = (i < cnn_predictions.size())
                ? cnn_predictions[i] + " (" + cv::format("%.2f", det.confidence) + ")"
                : "Unknown";

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int top = std::max(box.y - labelSize.height - baseLine, 0);
        int left = std::max(box.x, 0);
        cv::rectangle(image, cv::Point(left, top),
        cv::Point(left + labelSize.width, top + labelSize.height + baseLine),
        cv::Scalar(0, 255, 0), cv::FILLED);

        cv::putText(image, label, cv::Point(left, top + labelSize.height),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}
