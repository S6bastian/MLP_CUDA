// Data.cpp
#include "Data.h"
#include "MNIST_Reader.h"
#include <vector>
#include <cstdint>

using namespace std;

Data::Data(bool is_train) {
	if (is_train) {
		images = load_mnist_images("train-images.idx3-ubyte", num_images);
		labels = load_mnist_labels("train-labels.idx1-ubyte", num_labels);
	}
	else {
		images = load_mnist_images("t10k-images.idx3-ubyte", num_images);
		labels = load_mnist_labels("t10k-labels.idx1-ubyte", num_labels);
	}
}

vector<vector<float>> Data::get_normalized_images() const {
	vector<std::vector<float>> normalized;
	normalized.reserve(images.size());

	for (const auto& img : images) {
		vector<float> norm_img;
		norm_img.reserve(img.size());

		for (uint8_t pixel : img) {
			norm_img.push_back(static_cast<float>(pixel) / 255.0f);
		}

		normalized.push_back(move(norm_img));
	}

	return normalized;
}

vector<vector<float>> Data::get_one_hot_labels() const {
	vector<vector<float>> one_hot;
	one_hot.reserve(labels.size());

	for (uint8_t label : labels) {
		vector<float> oh(10, 0.0f);
		oh[label] = 1.0f;
		one_hot.push_back(move(oh));
	}

	return one_hot;
}
