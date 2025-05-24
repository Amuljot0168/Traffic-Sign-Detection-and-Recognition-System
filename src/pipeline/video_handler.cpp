#include "../include/TSDR/video_handler.h"
#include "../include/TSDR/draw.h"
#include "../config/config.h"
#include <opencv2/opencv.hpp>

VideoProcessor::VideoProcessor() : interface_engine(), frame_count(0), fps(0.0f) {
    // Initialize FPS timer to current time
    fps_timer_start = std::chrono::steady_clock::now();
}

void VideoProcessor::process(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Error opening video stream: " << video_path << std::endl;
            return;
        }

        cv::Mat frame;
        
        // Read frames until video ends or user quits
        while (cap.read(frame)) {
            frame_count++;

            // Run detection and recognition on every SKIP_FRAMES-th frame
            bool run_detection = (frame_count % SKIP_FRAMES == 0);
            if (run_detection) {
                 // Run detection+recognition pipeline and update predictions
                last_predictions = interface_engine.engine(frame, last_detections);
            }

            // Draw detection boxes and recognition labels on current frame
            draw_detections_with_cnn_predictions(frame, last_detections, last_predictions, CONF_THRESHOLD);

            // Update FPS counter every second
            update_fps();

            // Overlay FPS text on the frame
            display_fps(frame);

            // Show processed frame 
            cv::imshow("Video Prediction", frame);

            // Exit loop if ESC key is pressed
            if (cv::waitKey(1) == 27) break;
        }
}

void VideoProcessor::update_fps() {
    auto now = std::chrono::steady_clock::now();
        std::chrono::duration<float> elapsed = now - fps_timer_start;

        // Calculate FPS once every second
        if (elapsed.count() >= 1.0f) {
            fps = frame_count / elapsed.count();      // frames per second
            fps_timer_start = now;                   // reset timer
            frame_count = 0;                         // reset frame counter
        }
}

void VideoProcessor::display_fps(cv::Mat& frame) {
    std::stringstream ss;
        ss << "FPS: " << std::fixed << std::setprecision(2) << fps;

        // Put FPS text on frame at coordinates (10,30)
        cv::putText(frame, ss.str(), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
}