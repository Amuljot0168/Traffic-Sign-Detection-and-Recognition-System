#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "../include/TSDR/detection.h"
#include "../include/TSDR/recognition.h"
#include "../config/config.h"

// === Fixture Class ===
class DetectorTest : public ::testing::Test {
protected:
    Detector detector;

    DetectorTest()
        : detector(YOLO_MODEL_PATH, CONF_THRESHOLD, NMS_THRESHOLD) {}
};

// Tests that at least one valid detection is made on a known image
TEST_F(DetectorTest, ValidDetection) {
    cv::Mat image = cv::imread("../tests/test_data/stop_sign.jpg");
    ASSERT_FALSE(image.empty());

    std::vector<Detection> detections = detector.detect(image);

    EXPECT_GT(detections.size(), 0); 
    for (const auto& d : detections) {
        EXPECT_GE(d.confidence, 0.5);
        EXPECT_GT(d.box.area(), 0);
    }
}

// Tests that no detections are made on a blank image (edge case)
TEST_F(DetectorTest, NoDetectionOnEmptyImage) {
    cv::Mat blank = cv::Mat::zeros(cv::Size(416, 416), CV_8UC3);
    auto detections = detector.detect(blank);
    EXPECT_EQ(detections.size(), 0);
}

// Tests that cropped ROIs match the number of detections and are resized correctly
TEST_F(DetectorTest, ROICroppingMatchesDetection) {
    cv::Mat image = cv::imread("../tests/test_data/stop_sign.jpg");
    ASSERT_FALSE(image.empty());

    auto detections = detector.detect(image);
    auto crops = detector.crop_rois(image, detections);

    EXPECT_EQ(crops.size(), detections.size());
    for (const auto& crop : crops) {
        EXPECT_EQ(crop.size(), cv::Size(32, 32)); 
    }
}
