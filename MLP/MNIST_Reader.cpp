// MNIST_Reader.cpp
#include "MNIST_Reader.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

using namespace std;

vector<vector<uint8_t>> load_mnist_images(const string& path, int& num_images) {
	ifstream file(path, ios::binary);
	if (!file.is_open()) throw runtime_error("No se pudo abrir el archivo de imágenes.");
	
	int32_t magic_number = 0, rows = 0, cols = 0;
	file.read((char*)&magic_number, 4);
	file.read((char*)&num_images, 4);
	file.read((char*)&rows, 4);
	file.read((char*)&cols, 4);
	
	// Convertir de big endian a little endian (si es necesario)
	magic_number = _byteswap_ulong(magic_number);
	num_images = _byteswap_ulong(num_images);
	rows = _byteswap_ulong(rows);
	cols = _byteswap_ulong(cols);
	
	vector<vector<uint8_t>> images(num_images, vector<uint8_t>(rows * cols));
	
	for (int i = 0; i < num_images; ++i) {
		file.read((char*)images[i].data(), rows * cols);
	}
	
	return images;
}

vector<uint8_t> load_mnist_labels(const string& path, int& num_labels) {
	ifstream file(path, ios::binary);
	if (!file.is_open()) throw runtime_error("No se pudo abrir el archivo de etiquetas.");
	
	int32_t magic_number = 0;
	file.read((char*)&magic_number, 4);
	file.read((char*)&num_labels, 4);
	
	magic_number = _byteswap_ulong(magic_number);
	num_labels = _byteswap_ulong(num_labels);
	
	vector<uint8_t> labels(num_labels);
	file.read((char*)labels.data(), num_labels);
	
	return labels;
}

