#include "runner.h"
#include "../config/config.h"
#include "detection.h"
#include "recognition.h"
#include "../utils/draw.h"
#include "../utils/convert_cnn_labels.h"
// #include "../metrics/yolo_metrics.h"
// #include "../metrics/cnn_metrics.h"
// #include "../metrics/metrics.h"
#include <opencv2/opencv.hpp>

void process_images() {
    cv::dnn::Net yolo_net = load_yolo_model(YOLO_MODEL_PATH);
    cv::dnn::Net cnn_net = load_cnn_model(CNN_MODEL_PATH);
    cv::Mat image = cv::imread(IMAGE_PATH);
    if(image.empty())
        std::cerr << "Failed to load image. " << IMAGE_PATH << std::endl;
    
    auto yolo_outputs = run_yolo(image, yolo_net);
    auto detections = post_process_yolo(yolo_outputs[0], image.size());
    draw_detections(image, detections, CONF_THRESHOLD);
    // cv::imshow("Detection", image);
    // cv::waitKey(0);
    std::vector<cv::Mat> cropped_rois = crop_detection_roi(image, detections);
    std::vector<int> cnn_raw_predictions;
    for(auto& crop : cropped_rois) {
        cv::Mat preprocessed_crop = preprocess_images_for_cnn(crop);
        cv::Mat cnn_output = run_cnn(preprocessed_crop, cnn_net);
        cnn_raw_predictions.push_back(cnn_inference(cnn_output));
    }
    std::vector<std::string> cnn_predictions = get_cnn_labels_from_ids(cnn_raw_predictions);
    draw_detections_with_cnn_predictions(image, detections, cnn_predictions, CONF_THRESHOLD);
    cv::imshow("Prediction", image);
    cv::waitKey(0); 
}


