// Data.h
#ifndef DATA_H
#define DATA_H

#endif

#pragma once
#include <vector>
#include <cstdint>

using namespace std;

struct Data {
	int num_images, num_labels;
	std::vector<std::vector<uint8_t>> images;
	std::vector<uint8_t> labels;
	
	Data(bool is_train = true); // true: train, false: test
	vector<vector<float>> get_normalized_images() const;
	vector<vector<float>> get_one_hot_labels() const;
};
