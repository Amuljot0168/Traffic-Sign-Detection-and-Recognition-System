// // CNN label loading 
// vector<pair<string, int>> load_labels(const string& filename) {
//     vector<pair<string, int>> data;
//     ifstream file(filename);
//     string line;
//     string path;
//     int label;
    
//     if(!file.is_open()) {
//         cerr << "Could not open the label file. \n";
//         return data;
//     }
    
//     // skip header
//     getline(file, line);
    
//     while(getline(file, line)) {
//         stringstream ss(line);
//         getline(ss, path, ',');
//         ss >> label;
        
//         if(path.find('/') != string::npos)
//         path = path.substr(path.find_last_of('/') + 1);
        
//         data.emplace_back(path, label);
//     }
    
//     return (data);
// }

// // CNN inference
// int run_cnn_inference(Mat& blob) {
//     cnn_net.setInput(blob);
    
//     Mat output = cnn_net.forward();
//     Point class_id_point;
//     double confidence;
//     minMaxLoc(output, 0, &confidence, &class_id_point);

//     return (class_id_point.x);
// }

// CNN image pre processing





























// YOLO loading labels
// map<string, vector<Rect>> parse_ground_truth_files(const string& filepath, int YOLO_IMG_WIDTH = 640, int YOLO_IMG_HEIGHT = 640) {
//     map<string, vector<Rect>> labels;
//     ifstream infile(filepath);
//     string line;

//     while(getline(infile, line)) {
//         vector<string> parts;
//         stringstream ss(line);
//         string token;

//         while (getline(ss, token, ';')) {
//             parts.push_back(token);
//         }

//         if(parts.size() < 6)
//             continue;
        
//         string filename = parts[0];
//         float x_center = stof(parts[1]);
//         float y_center = stof(parts[2]);
//         float width = stof(parts[3]);
//         float height = stof(parts[4]);
//         int class_id = stoi(parts[5]); // optional

//         int box_x = int((x_center - width / 2.0) * YOLO_IMG_WIDTH);
//         int box_y = int((y_center - height / 2.0) * YOLO_IMG_HEIGHT);
//         int box_w = int(width * YOLO_IMG_WIDTH);
//         int box_h = int(height * YOLO_IMG_HEIGHT);

//         Rect bbox(box_x, box_y, box_w, box_h);
//         labels[filename].push_back(bbox);
//     }

//     return (labels);
// }

// YOLO image pre processing



// YOLO Interface running 


// YOLO post processing


// YOLO IOU calculation 
// float calculat_iou(const Rect& prediction, const Rect& truth) {
//     int x1 = max(prediction.x, truth.x);
//     int y1 = max(prediction.y, truth.y);
//     int x2 = min(prediction.x + prediction.width, truth.x + truth.width);
//     int y2 = min(prediction.y + prediction.height, truth.y + truth.height);
    
//     int intersection_area = max(0, x2 - x1) * max(0, y2 - y1);
//     int prediction_area = prediction.width * prediction.height;
//     int truth_area = truth.width * truth.height;

//     float iou = (float)intersection_area / (prediction_area + truth_area - intersection_area);

//     return (iou);
// }

// Evaluate YOLO 
// void evaluate_yolo_model(const string& image_dir, const string& label_file, Net& yolo_net) {
//     auto ground_truth = parse_ground_truth_files(label_file);
//     int total_predictions = 0;
//     int correct_predictions = 0;

//     for(const auto& [filename, gt_boxes] : ground_truth) {
//         string img_path = image_dir + "/" + filename;
//         Mat img = imread(img_path);

//         if(img.empty()) {
//             cerr << "Could not load image." << img_path << endl;
//             continue;
//         }

//         img = preprocess_yolo_input(img);
//         auto start = high_resolution_clock::now();
//         vector<Mat> outputs = run_yolo_interface(yolo_net, img);
//         vector<Rect> predicted_boxes = decode_yolo_output(outputs, img);
//         auto end = high_resolution_clock::now();

//         auto duration = duration_cast<microseconds>(end - start);
//         float fps = 1000000.0 / duration.count();

//         cout << "\n" << filename << " | " << predicted_boxes.size() << " boxes | Time: " << duration.count() << "μs | FPS: " << fps << endl;

//         total_predictions += predicted_boxes.size();

//         for(const auto& pred_box : predicted_boxes) {
//             for (const auto& gt_box : gt_boxes) {
//                 float iou = calculat_iou(pred_box, gt_box);
//                 if (iou >= 0.5f) {
//                     correct_predictions++;
//                     break;
//                 }
//             }
//         }
//     }

//     cout << "\nEvaluation Finished." << endl;
//     cout << "Correct Detections (IoU ≥ 0.5): " << correct_predictions << endl;
//     cout << "Total Predictions: " << total_predictions << endl;
//     cout << "Detection Accuracy: " << 100.0 * correct_predictions / total_predictions << "%" << endl;
// }
































// // Loading videox 
    // VideoCapture cap("D:/KPIT/opencvtest/data/video_input/bike.mp4");
    
    // if(!cap.isOpened()) {
    //     cerr << "Error opening video file. " << endl;
    //     return -1;
    // }

    // int frame_count = 0;
    // double total_infer_time = 0;

    // while(true) {
    //     Mat frame;
    //     cap >> frame;

    //     if(frame.empty())
    //         break;
        
    //     Mat input;
    //     resize(frame, input, Size(32, 32));
    //     input.convertTo(input, CV_32F, 1.0 / 255.0);

    //     Mat blob = blobFromImage(input);

    //     // Inference + timing
    //     auto start = chrono::high_resolution_clock::now();
    //     model.setInput(blob);

    //     Mat output = model.forward();

    //     auto end = chrono::high_resolution_clock::now();

    //     chrono::duration<double> elapsed = end - start;
    //     double infer_time_ms = elapsed.count() * 1000.0;
    //     total_infer_time += elapsed.count();
    //     frame_count++;

    //     // Getting Predictions
    //     Point class_id_point;
    //     double confidence;
    //     minMaxLoc(output, nullptr, &confidence, nullptr, &class_id_point);
    //     int class_id = class_id_point.x;

    //     cout << "Frame " << frame_count 
    //                     << " | Class: " << class_id 
    //                     << " | Confidence: " << confidence
    //                     << " | Inference Time: " << infer_time_ms << " ms" << endl;

    //     string label = class_id + " (" + to_string(confidence) + ")";
    //     putText(frame, label, Point(30, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

    //     imshow("Prediction", frame);
    //     if(waitKey(1) == 27)   // esc to exit
    //         break;
    // }

    // double avg_fps = frame_count / total_infer_time;
    // cout << "\nProcessed " << frame_count << " frames" << endl;
    // cout << "Average FPS: " << avg_fps << endl;

    // cap.release();
    // destroyAllWindows();
    // return 0;












    // vector<pair<string, int>> test_data = load_labels("D:/KPIT/opencvtest/data/mini_test_labels.csv");

    // int correct_predictions = 0;

    // for(const auto& [filename, true_label] : test_data) {
    //     string img_path = "D:/KPIT/opencvtest/data/Test/" + filename;
    //     Mat img = imread(img_path);

    //     if(img.empty()) {
    //         cerr << "Could not read image : " << img_path << endl;
    //         continue;
    //     }

    //     Mat blob = preprocess_images(img);

    //     auto start = high_resolution_clock::now();
    //     model.setInput(blob);
    //     Mat output = model.forward();

    //     auto end = high_resolution_clock::now();
    //     auto duration = duration_cast<microseconds>(end-start);
    //     float fps = 1000.0 / duration.count();

    //     cout << "Interference time: " << duration.count() << " ms, FPS: " << fps << endl;

    //     Point class_id_point;
    //     double confidence;
    //     minMaxLoc(output, 0, &confidence, 0, &class_id_point);
    //     int predicted = class_id_point.x;


    //     cout << "Image: " << filename << " | True: " << true_label << " | Prediced: " << predicted << endl;

    //     if(predicted == true_label)
    //         correct_predictions++;
    // }

    // cout << "\nAccuracy: " << (double)correct_predictions / test_data.size() * 100.0 << "%" << endl;


