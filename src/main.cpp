#include <opencv2/opencv.hpp>
#include "pipeline/runner.h"
#include "config/config.h"

int main(int argc, char** argv) {
    if(argc > 2 && std::string(argv[1]) == "--image") {
        std::string image_path = argv[2];
        process_images(image_path);
    }
    else if(argc > 2 && std::string(argv[1]) == "--video") {
        std::string video_path = argv[2];
        process_video(video_path);
    }
    else {
        std::cerr << "Usage: " << argv[0] << " --image <path_to_image> | --video <path_to_video>\n";
    }
    
    return 0;
}

// int main() {
//     process_images("D:/KPIT/opencvtest/test_data/YOLO_Test/images/00137.jpg");
//     process_video("D:/KPIT/opencvtest/test_data/video_input/night-c.mp4");
//     return 0;
// }