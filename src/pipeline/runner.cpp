#include "../include/TSDR/runner.h"
#include "../config/config.h"
#include "../include/TSDR/detection.h"
#include "../include/TSDR/recognition.h"
#include "../include/TSDR/draw.h"
#include "../include/TSDR/convert_cnn_labels.h"
#include <opencv2/opencv.hpp>

void process_images(std::string image_path) {
    cv::dnn::Net yolo_net = load_yolo_model(YOLO_MODEL_PATH);
    cv::dnn::Net cnn_net = load_cnn_model(CNN_MODEL_PATH);
    cv::Mat image = cv::imread(image_path);
    if(image.empty())
        std::cerr << "Failed to load image. " << image_path << std::endl;
    
    auto yolo_outputs = run_yolo(image, yolo_net);
    auto detections = post_process_yolo(yolo_outputs[0], image.size());
    draw_detections(image, detections, CONF_THRESHOLD);

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

void run(cv::Mat& frame, cv::dnn::Net& yolo_net, cv::dnn::Net& cnn_net, bool run_detection, std::vector<Detection>& last_detections, std::vector<std::string>& last_predictions) {
    std::vector<Detection> detections;

    if (run_detection) {
        auto yolo_outputs = run_yolo(frame, yolo_net);
        detections = post_process_yolo(yolo_outputs[0], frame.size());
        last_detections = detections;
        
        if (!detections.empty()) {
            std::vector<cv::Mat> cropped_rois = crop_detection_roi(frame, detections);
            std::vector<int> cnn_class_id_predictions;
            for (auto& crop : cropped_rois) {
                cv::Mat preprocessed_crop = preprocess_images_for_cnn(crop);
                cv::Mat cnn_output = run_cnn(preprocessed_crop, cnn_net);
                int prediction = cnn_inference(cnn_output);            
                cnn_class_id_predictions.push_back(prediction);
            }
        last_predictions = get_cnn_labels_from_ids(cnn_class_id_predictions);
        } else {
            last_predictions.clear();
        }
    }

    draw_detections_with_cnn_predictions(frame, last_detections, last_predictions, CONF_THRESHOLD);
}

void process_video(std::string video_path) {
    cv::dnn::Net yolo_net = load_yolo_model(YOLO_MODEL_PATH);
    cv::dnn::Net cnn_net = load_cnn_model(CNN_MODEL_PATH); 

    cv::Mat frame;
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream: " << video_path << std::endl;
        return;
    }

    auto fps_timer_start = std::chrono::steady_clock::now();
    int frame_count = 0;
    float fps = 0.0f;
    const int skip_frames = 5;
    std::vector<Detection> last_detections;
    std::vector<std::string> last_predictions;

    while (cap.read(frame)) {
        frame_count++;
        bool run_detection = (frame_count % skip_frames == 0);
        run(frame, yolo_net, cnn_net, run_detection, last_detections, last_predictions);
        
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<float> elapsed = now - fps_timer_start;

        if (elapsed.count() >= 1.0f) {
            fps = frame_count / elapsed.count();
            fps_timer_start = now;
            frame_count = 0;
        }

        std::stringstream fps_text;
        fps_text << "FPS: " << std::fixed << std::setprecision(2) << fps;
        cv::putText(frame, fps_text.str(), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Video", frame);
        if (cv::waitKey(1) == 27) break; // ESC to break
    }
    
}
