#include "include/yolo_utils.h"
#include "include/detection.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>

const float CONF_THRESHOLD = 0.25;
const float NMS_THRESHOLD = 0.45;
const float IOU_THRESHOLD = 0.5;
const int INPUT_WIDTH = 415;
const int INPUT_HEIGHT = 415;

int total_TP = 0;
int total_FP = 0;
int total_FN = 0;

// Function to load YOLO model
// cv::dnn::Net loadYoloModel(const std::string& model_path) {
//     cv::dnn::Net yolo_net = cv::dnn::readNetFromONNX(model_path);

//     std::cout << "Model loaded successfully from: " << model_path << std::endl;
//     std::vector<cv::String> input_names = yolo_net.getLayerNames();
//     std::cout << "Model has " << input_names.size() << " layers." << std::endl;

//     if (yolo_net.empty()) {
//         throw std::runtime_error("Failed to load the YOLO model");
//     }
//     return yolo_net;
// }

std::vector<cv::Rect> load_lables(const std::string& label_path, const cv::Size& image_size) {
    std::vector<cv::Rect> ground_truth;
    std::ifstream infile(label_path);
    
    if(!infile) {
        std::cerr << "Label file not found: " << label_path << std::endl;
        return ground_truth;
    }
    
    std::string line;
    while(getline(infile, line)) {
        std::cout << "Lable: " << line << std::endl;
        std::stringstream ss(line);
        int class_id;
        float x_center;
        float y_center;
        float width;
        float height;
        ss >> class_id >> x_center >> y_center >> width >> height;

        int img_width = image_size.width;
        int img_height = image_size.height;

        int xmin = static_cast<int>((x_center - width / 2) * img_width);
        int ymin = static_cast<int>((y_center - height / 2) * img_height);
        int xmax = static_cast<int>((x_center + width / 2) * img_width);
        int ymax = static_cast<int>((y_center + height / 2) * img_height);

        xmin = std::max(0, std::min(xmin, img_width - 1));
        ymin = std::max(0, std::min(ymin, img_height - 1));
        xmax = std::max(0, std::min(xmax, img_width - 1));
        ymax = std::max(0, std::min(ymax, img_height - 1));

        ground_truth.push_back(cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin));
    }

    return ground_truth;
}

// std::vector<Detection> post_process(const cv::Mat& output, const cv::Size& image_size) {
//     std::vector<Detection> detections;
//     std::cout << "Output shape: " << output.size << std::endl;
//     int rows = output.size[1];
//     int dimensions = output.size[2];
//     const float* data = (float*)output.data;
    
//     for(int i = 0; i < rows; ++i) {
//         float object_conf = data[i * dimensions + 4];
//         if(object_conf < CONF_THRESHOLD)
//             continue;
        
//         float max_class_score = -1.0f;
//         int class_id = -1;
//         for(int j  = 5; j < dimensions; ++j) {
//             float class_score = data[i * dimensions + j];
//             if(class_score > max_class_score) {
//                 max_class_score = class_score;
//                 class_id = j - 5;
//             }
//         }
            
//         float confidence = object_conf * max_class_score;
//         if(confidence < CONF_THRESHOLD)
//         continue;
            
//         float cx = data[i * dimensions + 0];
//         float cy = data[i * dimensions + 1];
//         float  w = data[i * dimensions + 2];
//         float  h = data[i * dimensions + 3];
//         float x = cx - w / 2.0f;
//         float y = cy - h / 2.0f;
//         float scale_x = (float)image_size.width / 415.0f;
//         float scale_y = (float)image_size.height / 415.0f;

//         int left = int(x * scale_x);
//         int top = int(y * scale_y);
//         int width = int(w * scale_x);
//         int height = int(h * scale_y);

//         cv::Rect box(left, top, width, height);

//         detections.push_back({ class_id, confidence, box });
//     }
        
//     std::vector<int> indices;
//     std::vector<cv::Rect> boxes;
//     std::vector<float> scores;
        
//     for(const auto& d: detections) {
//         boxes.push_back(d.box);
//         scores.push_back(d.confidence);
//     }
        
//     std::vector<Detection> final_detections;
//     cv::dnn::NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD, indices);
    
//     for(int idx : indices)
//     final_detections.push_back(detections[idx]);
        
//     return final_detections;
// }

// void draw_detections(const cv::Mat& image, const std::vector<Detection>& detections, float CONF_THRESHOLD) {
//     for (const auto& det : detections) {
//         if (det.confidence < CONF_THRESHOLD) {
//             continue;
//         }

//         cv::Rect box = det.box & cv::Rect(0, 0, image.cols, image.rows);
//         if (box.width <= 0 || box.height <= 0) {
//             std::cerr << "Warning: Invalid box dimensions: " << box << std::endl;
//             continue;
//         }

//         rectangle(image, box, cv::Scalar(0, 255, 0), 2);

//         std::string label = "ID:" + std::to_string(det.class_id) + " (" + cv::format("%.2f", det.confidence) + ")";
//         int baseLine = 0;
//         cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

//         // Ensure text background fits and doesn't go out of image bounds
//         int top = std::max(box.y - labelSize.height - baseLine, 0);
//         int left = std::max(box.x, 0);
//         cv::rectangle(image, cv::Point(left, top),
//                       cv::Point(left + labelSize.width, top + labelSize.height + baseLine),
//                       cv::Scalar(0, 255, 0), cv::FILLED);

//         // Draw label
//         cv::putText(image, label, cv::Point(left, top + labelSize.height),
//                     cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
//     }
// }

float calculateIoU(const cv::Rect& pred_box, const cv::Rect& true_box) {
    int x_left = cv::max(pred_box.x, true_box.x);
    int y_top = cv::max(pred_box.y, true_box.y);
    int x_right = cv::min(pred_box.x + pred_box.width, true_box.x + true_box.width);
    int y_bottom = cv::min(pred_box.y + pred_box.height, true_box.y + true_box.height);

    int intersection_area = cv::max(0, x_right - x_left) * cv::max(0, y_bottom - y_top);

    int pred_area = pred_box.width * pred_box.height;
    int true_area = true_box.width * true_box.height;

    int union_area = pred_area + true_area - intersection_area;

    float iou = (float)intersection_area / union_area;
    return iou;
}

void evaluateIoU(const std::vector<Detection>& detections, const std::vector<cv::Rect>& ground_truths, int& total_TP, int& total_FP, int& total_FN, float IOU_THRESHOLD) {
    int TP = 0;
    int FP = 0;
    std::vector<bool> matched(ground_truths.size(), false);

    for (const auto& detection : detections) {
        bool match_found = false;
        float best_iou = 0.0f;
        int best_idx = -1;

        for (size_t i = 0; i < ground_truths.size(); ++i) {
            float iou = calculateIoU(detection.box, ground_truths[i]);
            if (iou > best_iou && !matched[i]) {
                best_iou = iou;
                best_idx = i;
            }
        }

        if (best_iou >= IOU_THRESHOLD && best_idx != -1) {
            match_found = true;
            matched[best_idx] = true;
            TP++;
        }
        else {
            FP++;
        }
    }

    int FN = ground_truths.size() - TP;

    total_TP += TP;
    total_FP += FP;
    total_FN += FN;
}

std::vector<cv::Mat> run_yolo(const cv::Mat& img, cv::dnn::Net& yolo_net, std::string label_path) {
    const int input_width = 415;
    const int input_height = 415;

    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob, 1.0/255.0, cv::Size(input_width, input_height), cv::Scalar(), true, false);
    yolo_net.setInput(blob);

    std::vector<cv::Mat> outputs;
    yolo_net.forward(outputs, yolo_net.getUnconnectedOutLayersNames());

    // std::cout << "Processed Image, detections shape: " << outputs[0].size << std::endl;

    // std::vector<Detection> detections = post_process(outputs[0], img.size());
    // std::cout << "Number of Detections: " << detections.size() << std::endl;
    // draw_detections(img, detections, 0.5);
    // imshow("Detections", img);
    // cv::waitKey(0);

    // std::vector<cv::Rect>ground_truth = load_lables(label_path, img.size());
    // evaluateIoU(detections, ground_truth, total_TP, total_FP, total_FN, IOU_THRESHOLD);
    return outputs;
}

// Function to calculate and print final performance metrics
void calculateFinalMetrics(int total_TP, int total_FP, int total_FN) {
    float precision = (total_TP + total_FP) > 0 ? (float)total_TP / (total_TP + total_FP) : 0.0f;
    float recall = (total_TP + total_FN) > 0 ? (float)total_TP / (total_TP + total_FN) : 0.0f;
    float f1_score = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0f;

    std::cout << "\n=== Final Metrics ===\n";
    std::cout << "Precision: " << precision << "\n";
    std::cout << "Recall: " << recall << "\n";
    std::cout << "F1 Score: " << f1_score << "\n";
    std::cout << "TP: " << total_TP << " | FP: " << total_FP << " | FN: " << total_FN << "\n";
}
