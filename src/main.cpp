#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <chrono>
using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace std::chrono;

const int YOLO_IMG_WIDTH = 416;
const int YOLO_IMG_HEIGHT = 416;
const int CNN_IMG_WIDTH = 32;
const int CNN_IMG_HEIGHT = 32;
// float YOLO_CONFIDENCE_THRESHOLD = 0.50;     usual approach
float YOLO_CONFIDENCE_THRESHOLD = 0.40;
// float YOLO_NMS_THRESHOLD = 0.45;            usual threshold
float YOLO_NMS_THRESHOLD = 0.40;


Net yolo_net = readNetFromONNX("D:/KPIT/opencvtest/models/v5-4-class.onnx");
Net cnn_net = readNetFromONNX("D:/KPIT/opencvtest/models/cnn_recognition_model_nchw.onnx");

unordered_map<int, int> gtsrb_to_gtsdb_mapping = {
    {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}, 
    {10, 10}, {11, 11}, {12, 12}, {13, 13}, {14, 14}, {15, 15}, {16, 16}, {17, 17}, {18, 18}, {19, 19}, 
    {20, 20}, {21, 21}, {22, 22}, {23, 23}, {24, 24}, {25, 25}, {26, 26}, {27, 27}, {28, 28}, 
    {30, 31}, {32, 40}
};

typedef struct GroundTruth {
    string image_name;
    int class_id;
    Rect box;

} GroundTruth;

typedef struct RawBox {
    int class_id;
    float confidence;
    Rect box;
} RawBox;


vector<GroundTruth> load_ground_truth(const string& filepath) {
    vector<GroundTruth> ground_truth_data;
    ifstream file(filepath);
    string line;

    if(!file.is_open()) {
        cerr << "Error opening file: " << filepath << endl;
        return ground_truth_data;
    }

    while(getline(file, line)) {
        stringstream ss(line);
        string img_name;
        int x;
        int y;
        int w;
        int h;
        int id;

        getline(ss, img_name, ';');
        ss >> x; ss.ignore();
        ss >> y; ss.ignore();
        ss >> w; ss.ignore();
        ss >> h; ss.ignore();
        ss >> id; 

        Rect box = Rect(x, y, w, h);
        ground_truth_data.push_back(GroundTruth{img_name, id, box});
    }
    file.close();

    return (ground_truth_data);
}

Mat yolo_preprocess_img(const Mat& img) {
    cout << "Preprocessing image for yolo. " << endl;
    Mat resized_img;
    resize(img, resized_img, Size(416, 416));    // resize check 
    
    resized_img.convertTo(resized_img, CV_32F, 1.0 / 255.0);
    cvtColor(resized_img, resized_img, COLOR_BGR2RGB);
    
    return (resized_img);
}

vector<Mat> run_yolo(const Mat& img, Net& yolo_net) {
    cout << "Running YOLO. " << endl;
    Mat input_blob = blobFromImage(img, 1.0, Size(416, 416), Scalar(0, 0, 0), true, false);
    yolo_net.setInput(input_blob);
    
    vector<Mat> outputs;
    yolo_net.forward(outputs, yolo_net.getUnconnectedOutLayersNames()); 
    return (outputs);
}

vector<RawBox> extract_bounding_boxes(const Mat& output, const Size& original_size, float confidence_threshold = YOLO_CONFIDENCE_THRESHOLD) {
    cout << "Extracting bounding boxes. " << endl;
    vector<RawBox> boxes;

    // cout << "dims: " << output.dims << endl;
    // for (int i = 0; i < output.dims; ++i) {
    //     cout << "size[" << i << "]: " << output.size[i] << endl;
    // }
    Mat reshaped_output = output.reshape(1, output.size[1]); // from (1, 25200, 48) → (25200, 48)

    int rows = reshaped_output.rows;
    cout << "Output rows: " << rows << endl;
    int cols = reshaped_output.cols;
    cout << "Output cols: " << cols << endl;
    
    for(int i = 0; i < rows; ++i) { 
        const float* data = reshaped_output.ptr<float>(i);
        float objectness = data[4];

        if (i < 5) {  // Just print first 5 rows
            cout << "Row " << i << " objectness score: " << objectness << endl;
        }

        if(objectness < confidence_threshold) 
            continue;

        float max_class_score = -1;
        int class_id = -1;
        for(int j = 5; j < cols; ++j) {
            if(data[j] > max_class_score) {
                max_class_score = data[j];
                class_id = j - 5;
            }
        }

        float confidence = objectness * max_class_score;
        if(confidence < confidence_threshold)
            continue;

        float x = data[0];
        float y = data[1];
        float w = data[2];
        float h = data[3];

        float x_scale = static_cast<float>(original_size.width) / YOLO_IMG_WIDTH;
        float y_scale = static_cast<float>(original_size.height) / YOLO_IMG_HEIGHT;
    
        int left = static_cast<int>((x - w / 2.0f) * x_scale);
        int top = static_cast<int>((y - h / 2.0f) * y_scale);
        int width = static_cast<int>(w * x_scale);
        int height = static_cast<int>(h * y_scale);

        Rect rect = Rect(left, top, width, height);

        boxes.push_back({class_id, confidence, rect});
    }
    
    return (boxes);
}

float calculate_iou(const Rect& prediction, const Rect& truth) {
    cout << "Calculating IOU. " << endl;
    int intersection_area = (prediction & truth).area();
    int union_area = prediction.area() + truth.area() - intersection_area;

    return (union_area > 0 ? static_cast<float>(intersection_area) / union_area : 0.0f);
}

vector<RawBox> validate_yolo_predition_with_iou(const vector<RawBox>& predictions, vector<GroundTruth> ground_truth, string image_name, float iou_threshold = 0.5) {
    cout << "Validating yolo prediction with iou. " << endl;
    vector<RawBox> validate;

    for(const auto& pred : predictions) {
        for(const auto& gt : ground_truth) {
            if(gt.image_name != image_name)
                continue;

            float iou = calculate_iou(pred.box, gt.box);
            if(iou >= iou_threshold && pred.class_id == gt.class_id) {
                validate.push_back(pred);
                break;
            }
        }
    }
    return validate;
}

vector<Mat> cropped_validated_boxes(const Mat& image, const vector<RawBox>& boxes) {
    cout << "Cropping validated boxes. " << endl;
    vector<Mat> cropped_rois;
    
    for(const auto& box : boxes) {
        Rect valid_box = box.box & Rect(0, 0, image.cols, image.rows);

        if (valid_box.width <= 0 || valid_box.height <= 0) {
            continue; // skip invalid boxes
        }

        if(valid_box.area() > 0) {
            Mat cropped = image(valid_box).clone();
            cropped_rois.push_back(cropped);
        }
        cout << "Cropping ROI: " << box.box << " from image size: "
         << image.cols << "x" << image.rows << endl;
    }


    return cropped_rois;
}

vector<Mat> preprocess_images_for_cnn(const vector<Mat>& cropped_rois) {
    cout << "Preprocessing cropped images for CNN" << endl;
    vector<Mat> processed_images;

    for(const auto& roi : cropped_rois) {
        if(roi.empty())
            continue;
        
        Mat resized;
        cvtColor(roi, roi, COLOR_BGR2RGB);
        resize(roi, resized, Size(32, 32));
        
        Mat blob;
        blobFromImage(resized, blob, 1.0 / 255.0, Size(32, 32), Scalar(), true, false);

        processed_images.push_back(blob);
    }
    return (processed_images);
}

int run_cnn_inference(Mat& blob) {
    cout << "Running CNN. " << endl;
    cnn_net.setInput(blob);
    
    Mat output = cnn_net.forward();
    Point class_id_point;
    double confidence;
    minMaxLoc(output, nullptr, &confidence, nullptr, &class_id_point);
    int class_id = class_id_point.x;

    return (class_id_point.x);
}

int map_cnn_prediction_to_gtsdb(int cnn_prediction) {
    cout << "Mapping CNN predictions to GTSDB. " << endl;
    if(gtsrb_to_gtsdb_mapping.find(cnn_prediction) != gtsrb_to_gtsdb_mapping.end()) {
        return gtsrb_to_gtsdb_mapping[cnn_prediction];
    }
    else {
        return -1;
    }
}

bool validate_prediction(int cnn_mapped_prediction, int cnn_ground_truth) {
    cout << "Validating CNN preictions. " << endl;
    return cnn_mapped_prediction == cnn_ground_truth;
}


void update_metrics(bool cnn_is_correct, int& total_predictons, int& cnn_correct_predictions) {
    total_predictons++;
    if(cnn_is_correct) {
        cnn_correct_predictions++;
    }
}

// int argc, char** argv 
int main() {   
    cout << "Inside main" << endl;
    string image_dir = "D:/KPIT/opencvtest/data/yolo_jpg/"; // e.g., /content/images
    string label_file = "D:/KPIT/opencvtest/data/mini_yolo.txt";

    int total_predictions = 0;
    int cnn_correct_predictions = 0;

    auto ground_truth_data = load_ground_truth(label_file);
    if(!ground_truth_data.empty())
        cout << "Ground truth loaded successfully. " << endl;
    
    for(const auto& gt : ground_truth_data) {
        string img_filename = gt.image_name;
        size_t dot_pos = img_filename.find_last_of(".");
        if (dot_pos != string::npos) {
            img_filename = img_filename.substr(0, dot_pos) + ".jpg";
        }
        string img_path = image_dir + img_filename;
        Mat img = imread(img_path);

        if(img.empty()) {
            cerr << "Failed to load image " << img_path << endl;
            continue;
        }
        else 
        cout << "Image" << img_path << " loaded sucecssfully. " << endl;

        Size original_size = img.size();

        Mat input_blob = yolo_preprocess_img(img);

        auto yolo_outputs = run_yolo(input_blob, yolo_net);
        
        // for (int i = 0; i < yolo_outputs.size(); ++i) {
        //     cout << "Output " << i << " shape: " << yolo_outputs[i].size << endl;
        // }

        
        // if(!yolo_outputs.empty())
        //     cout << "Outputs generated." << endl;

        vector<RawBox> raw_predictions;
        
        for(const auto& output : yolo_outputs) {
            auto extracted = extract_bounding_boxes(output, original_size);
            raw_predictions.insert(raw_predictions.end(), extracted.begin(), extracted.end());
        }

        if(!raw_predictions.empty()) 
            cout << "Raw predictions extracted successfully. " << endl;

        // auto validate_boxes = validate_yolo_predition_with_iou(raw_predictions, ground_truth_data, gt.image_name);
        auto validate_boxes = raw_predictions; 
        cout << "Validated boxes count: " << validate_boxes.size() << endl;

        
        auto cropped_rois = cropped_validated_boxes(img, validate_boxes);
        cout << "Cropped ROIs count: " << cropped_rois.size() << endl;

        auto cnn_inputs = preprocess_images_for_cnn(cropped_rois);

        for(size_t i = 0; i < cnn_inputs.size(); ++i) {
            cout << "INSIDE" << endl;
            if (cnn_inputs[i].empty() || cnn_inputs[i].cols <= 1 || cnn_inputs[i].rows <= 1) {
                cerr << "Skipping invalid ROI - empty or too small" << endl;
                continue;
            }
            cout << "CNN input size " << i << ": " << cnn_inputs[i].size;
            
            int cnn_prediction = run_cnn_inference(cnn_inputs[i]);
            int mapped_prediction = map_cnn_prediction_to_gtsdb(cnn_prediction);

            bool is_correct = validate_prediction(mapped_prediction, gt.class_id);
            update_metrics(is_correct, total_predictions, cnn_correct_predictions);
        }

    }

    cout << "CNN Accuracy: " << (total_predictions > 0 ? 100.0 * cnn_correct_predictions / total_predictions : 0.0) << "%" << endl;
    cout << "Total Predictions: " << total_predictions << ", Correct Predictions: " << cnn_correct_predictions << endl;

    return 0;
}
