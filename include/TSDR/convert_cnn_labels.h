#ifndef CONVERT_CNN_LABELS_H
#define CONVERT_CNN_LABELS_H
#include <string>
#include <vector>
#include <unordered_map>

extern const std::unordered_map<int, std::string> label_map;

std::vector<std::string> get_cnn_labels_from_ids(const std::vector<int>& cnn_class_ids);
#endif