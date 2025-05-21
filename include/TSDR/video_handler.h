#ifndef VIDEO_HANDLER_H
#define VIDEO_HANDLER_H

#include "interface_engine.h"
#include <opencv2/opencv.hpp>

class VideoProcessor {
public: 
    VideoProcessor();

    void process(const std::string& video_path);

private: 
    InterfaceEngine interface_engine;
    int skip_frames;
    int frame_count;
    float fps;
    std::chrono::steady_clock::time_point fps_timer_start;
    std::vector<Detection> last_detections;
    std::vector<std::string> last_predictions;

    void update_fps();
    void display_fps(cv::Mat& frame);
};

#endif