#include "../include/TSDR/video_handler.h"
#include "../include/TSDR/draw.h"
#include "../config/config.h"
#include <opencv2/opencv.hpp>

VideoProcessor::VideoProcessor() : interface_engine(), frame_count(0), fps(0.0f) {
    fps_timer_start = std::chrono::steady_clock::now();
}

void VideoProcessor::process(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Error opening video stream: " << video_path << std::endl;
            return;
        }

        cv::Mat frame;
        while (cap.read(frame)) {
            frame_count++;

            bool run_detection = (frame_count % SKIP_FRAMES == 0);
            if (run_detection) {
                last_predictions = interface_engine.engine(frame, last_detections);
            }

            draw_detections_with_cnn_predictions(frame, last_detections, last_predictions, CONF_THRESHOLD);

            update_fps();

            display_fps(frame);

            cv::imshow("Video Prediction", frame);
            if (cv::waitKey(1) == 27) break;  // ESC to quit
        }
}

void VideoProcessor::update_fps() {
    auto now = std::chrono::steady_clock::now();
        std::chrono::duration<float> elapsed = now - fps_timer_start;
        if (elapsed.count() >= 1.0f) {
            fps = frame_count / elapsed.count();
            fps_timer_start = now;
            frame_count = 0;
        }
}

void VideoProcessor::display_fps(cv::Mat& frame) {
    std::stringstream ss;
        ss << "FPS: " << std::fixed << std::setprecision(2) << fps;
        cv::putText(frame, ss.str(), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
}