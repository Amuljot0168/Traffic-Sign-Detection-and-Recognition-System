#ifndef INTERFACE_ENGINE_H
#define INTERFACE_ENGINE_H

#include "detection.h"
#include "recognition.h"

class InterfaceEngine {
public: 
    InterfaceEngine();
    std::vector<std::string> engine(const cv::Mat& frame, std::vector<Detection>& detections);

private:
    Detector yolo;
    Classifier cnn;
};

#endif