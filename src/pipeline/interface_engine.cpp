#include "../include/TSDR/interface_engine.h"
#include "../config/config.h"
#include "../include/TSDR/convert_cnn_labels.h"

InterfaceEngine::InterfaceEngine() : yolo(YOLO_MODEL_PATH, CONF_THRESHOLD, NMS_THRESHOLD, cv::Size(32,32)), cnn(CNN_MODEL_PATH) {}

std::vector<std::string> InterfaceEngine::engine(const cv::Mat& frame, std::vector<Detection>& detections) {
    detections = yolo.detect(frame);
        if (detections.empty()) return {};

        auto cropped_rois = yolo.crop_rois(frame, detections);

        std::vector<int> class_ids;
        for (auto& roi : cropped_rois) {
            class_ids.push_back(cnn.predict(roi));
        }

        return get_cnn_labels_from_ids(class_ids);
}