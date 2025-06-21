// MNIST_Reader.h
#pragma once
#include <vector>
#include <string>
#include <cstdint>

std::vector<std::vector<uint8_t>> load_mnist_images(const std::string& path, int& num_images);
std::vector<uint8_t> load_mnist_labels(const std::string& path, int& num_labels);
