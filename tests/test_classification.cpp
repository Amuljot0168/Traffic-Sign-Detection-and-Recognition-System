#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "../include/TSDR/detection.h"
#include "../include/TSDR/recognition.h"
#include "../config/config.h"


// === Fixture Class ===
class ClassifierTest : public ::testing::Test {
protected:
    Classifier classifier;

    ClassifierTest()
        : classifier(CNN_MODEL_PATH, 32) {}
};

// Tests if preprocessing resizes input to 32x32 as expected
TEST_F(ClassifierTest, preprocessing_resizes_to_expected_dimensions) {
    // Classifier classifier(cnn_model_path, 32);

    cv::Mat image = cv::imread("../tests/test_data/stop_sign.jpg");
    ASSERT_FALSE(image.empty());

    cv::Mat blob = classifier.preprocess(image);
    EXPECT_EQ(blob.size[2], 32);
    EXPECT_EQ(blob.size[3], 32);
}

// Tests if prediction correctly classifies a known stop sign image
TEST_F(ClassifierTest, predicts_known_label_correctly) {
    // Classifier classifier(cnn_model_path, 32);

    cv::Mat crop = cv::imread("../tests/test_data/stop_sign_crop.png");
    ASSERT_FALSE(crop.empty());

    int label = classifier.predict(crop);
    EXPECT_EQ(label, 1);  // Valid class ID
}

// Tests if predict() handles empty input images gracefully
TEST_F(ClassifierTest, handles_empty_input_gracefully) {
    // Classifier classifier(cnn_model_path, 32);
    
    cv::Mat empty;
    try {
        int label = classifier.predict(empty);
        FAIL() << "Should throw or handle empty image";
    } catch (const std::exception& e) {
        SUCCEED();
    } catch (...) {
        FAIL() << "Unknown exception thrown";
    }
}