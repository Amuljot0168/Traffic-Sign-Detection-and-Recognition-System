#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>


cv::dnn::Net cnn_net = cv::dnn::readNetFromONNX("D:/KPIT/opencvtest/models/cnn_recognition_model_nchw.onnx");

std::vector<std::pair<std::string, int>> load_labels(const std::string& filename) {
    std::vector<std::pair<std::string, int>> data;
    std::ifstream file(filename);
    std::string line;
    std::string path;
    int label;
    
    if(!file.is_open()) {
        std::cerr << "Could not open the label file. \n";
        return data;
    }
    
    // skip header
    getline(file, line);
    
    while(getline(file, line)) {
        std::stringstream ss(line);
        getline(ss, path, ',');
        ss >> label;
        
        if(path.find('/') != std::string::npos)
        path = path.substr(path.find_last_of('/') + 1);
        
        data.emplace_back(path, label);
    }
    
    return (data);
}

cv::Mat preprocess_images(const cv::Mat& img) {
    cv::Mat resized, blob;
    cv::resize(img, resized, cv::Size(32, 32));

    blob = cv::dnn::blobFromImage(resized, 1.0 / 255.0, cv::Size(32, 32), cv::Scalar(), false, false);

    return blob;
}


int main() {
    std::vector<std::pair<std::string, int>> test_data = load_labels("D:/KPIT/opencvtest/data/mini_test_labels.csv");

    int correct_predictions = 0;

    for(const auto& [filename, true_label] : test_data) {
        std::string img_path = "D:/KPIT/opencvtest/data/CNN_Test/" + filename;
        cv::Mat img = cv::imread(img_path);

        if(img.empty()) {
            std::cerr << "Could not read image : " << img_path << std::endl;
            continue;
        }

        cv::Mat blob = preprocess_images(img);

        auto start = std::chrono::high_resolution_clock::now();
        cnn_net.setInput(blob);
        cv::Mat output = cnn_net.forward();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
        float fps = 1000.0 / duration.count();

        std::cout << "Interference time: " << duration.count() << " ms, FPS: " << fps << std::endl;

        cv::Point class_id_point;
        double confidence;
        minMaxLoc(output, 0, &confidence, 0, &class_id_point);
        int predicted = class_id_point.x;


        std::cout << "Image: " << filename << " | True: " << true_label << " | Prediced: " << predicted << std::endl;

        if(predicted == true_label)
            correct_predictions++;
    }

    std::cout << "\nAccuracy: " << (double)correct_predictions / test_data.size() * 100.0 << "%" << std::endl;

    return 0;
}