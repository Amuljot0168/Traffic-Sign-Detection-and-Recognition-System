#include "../include/TSDR/interface_engine.h"
#include "../config/config.h"
#include "../include/TSDR/convert_cnn_labels.h"

InterfaceEngine::InterfaceEngine() : yolo(YOLO_MODEL_PATH, CONF_THRESHOLD, NMS_THRESHOLD, cv::Size(32,32)), cnn(CNN_MODEL_PATH) {}

std::vector<std::string> InterfaceEngine::engine(const cv::Mat& frame, std::vector<Detection>& detections) {
    // Run YOLO detection on the input frame
    detections = yolo.detect(frame);

     // If no detections, return empty vector
    if (detections.empty()) return {};

    // Crop regions of interest (ROIs) from the frame based on detections for CNN input
    auto cropped_rois = yolo.crop_rois(frame, detections);

     // Predict class IDs for each cropped ROI using the CNN
    std::vector<int> class_ids;
    for (auto& roi : cropped_rois) {
        class_ids.push_back(cnn.predict(roi));
    }

    // Convert class IDs to human-readable labels and return
    return get_cnn_labels_from_ids(class_ids);
}