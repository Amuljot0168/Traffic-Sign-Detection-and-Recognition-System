#include <opencv2/opencv.hpp>
#include "include/TSDR/image_handler.h"
#include "include/TSDR/video_handler.h"
#include "config/config.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage:\n"
                  << argv[0] << " --image <image_path>\n"
                  << argv[0] << " --video <video_path>\n";
        return 1;
    }

    if(argc > 2) {
        std::string mode = argv[1];
        std::string path = argv[2];
    
        if (mode == "--image") {
            ImageProcessor ip;
            ip.process(path);
        } else if (mode == "--video") {
            VideoProcessor vp;
            vp.process(path);
        } 
        else {
            std::cerr << "Usage: " << argv[0] << " --image <path_to_image> | --video <path_to_video>\n";
            return 1;
        }
    }

    return 0;
}
