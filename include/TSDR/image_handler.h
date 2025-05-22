#ifndef IMAGE_HANDLER_H
#define IMAGE_HANDLER_H

#include "interface_engine.h"
#include <opencv2/opencv.hpp>

class ImageProcessor {
public: 
    ImageProcessor();

    void process(const std::string& image_path);

private: 
    InterfaceEngine interface_engine;
};

#endif
