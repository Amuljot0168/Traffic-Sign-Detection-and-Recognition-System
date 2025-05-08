#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <chrono>
#include "include/yolo_utils.h"
#include "include/detection.h"

int main() {
    std::cout << "opencv version" << CV_VERSION << std::endl;
    int TP = 0;
    int FP = 0;
    int FN = 0;

    // bool use_video = false;
    // std::string input_path = "path/to/video_or_image";
    
    // if (argc > 1 && std::string(argv[1]) == "--video") {
    //     use_video = true;
    // }

    // Model loading
    std::string model_path = "D:/KPIT/opencvtest/models/v5-4-class.onnx";
    cv::dnn::Net yolo_net = loadYoloModel(model_path);

    
    
    // cv::Mat frame;  // This is your input to the detection function

    // if (use_video) {
    //     // Video mode: loop through frames
    //     cv::VideoCapture cap(video_path);
    //     while (cap.read(frame)) {
    //         // ↓ Same functions as used for image ↓
    //         auto detections = runDetection(frame, net);
    //         drawDetections(frame, detections);
    //         cv::imshow("Video", frame);
    //         if (cv::waitKey(1) == 27) break;
    //     }
    // } else {
    //     // Image mode: load once
    //     frame = cv::imread(image_path);
    //     auto detections = runDetection(frame, net);
    //     drawDetections(frame, detections);
    //     cv::imshow("Image", frame);
    //     cv::waitKey(0);
    // }

    // Testing logic
    std::string image_dir = "D:/KPIT/opencvtest/data/YOLO_Test/images/";
    std::string labels_dir = "D:/KPIT/opencvtest/data/YOLO_Test/labels/";

    for(const auto &entry: std::filesystem::directory_iterator(image_dir)) {
        std::string img_path = entry.path().string();
        std::string filename = entry.path().stem().string();
        std::string label_path = labels_dir + filename + ".txt";

        std::cout << "Processing image: " << img_path << std::endl;

        cv::Mat img = cv::imread(img_path);
        if(img.empty()) {
            std::cerr << "Could not load image: " << img_path << std::endl;
            continue;
        }
        std::vector<cv::Mat> outputs = run_yolo(img, yolo_net, label_path);

        std::vector<Detection> detections = post_process(outputs[0], img.size());
        draw_detections(img, detections, CONF_THRESHOLD);
        imshow("Detections", img);
        cv::waitKey(0);

        std::vector<cv::Rect> ground_truth = load_lables(label_path, img.size());
        evaluateIoU(detections, ground_truth, total_TP, total_FP, total_FN, IOU_THRESHOLD);
        

    }
    calculateFinalMetrics(total_TP, total_FP, total_FN);
    return 0;
}