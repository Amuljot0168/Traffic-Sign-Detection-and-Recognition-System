#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "../include/TSDR/detection.h"
#include "../include/TSDR/recognition.h"
#include "../config/config.h"

const std::string yolo_model_path = YOLO_MODEL_PATH;  
const std::string cnn_model_path = CNN_MODEL_PATH;

// Tests end-to-end detection and recognition pipeline
TEST(SystemTests, end_to_end_detection_and_recognition) {
    Detector detector(yolo_model_path, 0.5f, 0.4f);
    Classifier classifier(cnn_model_path, 32);
    
    cv::Mat frame = cv::imread("../tests/test_data/multi_signs.jpg");
    ASSERT_FALSE(frame.empty());

    auto detections = detector.detect(frame);
    auto crops = detector.crop_rois(frame, detections);

    ASSERT_EQ(detections.size(), crops.size());

    for (const auto& crop : crops) {
        int label = classifier.predict(crop);
        EXPECT_GE(label, 0);
    }
}
