// Data.h
#pragma once
#include <vector>
#include <cstdint>

struct Data {
    int num_images, num_labels;
    std::vector<std::vector<uint8_t>> images;
    std::vector<uint8_t> labels;

    Data(bool is_train = true);
    std::vector<std::vector<float>> get_normalized_images() const;
    std::vector<std::vector<float>> get_one_hot_labels() const;
};