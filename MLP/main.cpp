// main.cpp
#include "MNIST_Reader.h"
#include "Data.h"
#include "mlp_cuda_types.h" // Include the new CUDA MLP header
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>

using namespace std;

// Guardar loss por época
void save_loss_log(const vector<float>& losses, const string& filename = "loss_log.txt") {
	ofstream file(filename);
	if (!file.is_open()) {
		cerr << "Error al abrir " << filename << endl;
		return;
	}
	file << "Epoch,Loss\n";
	for (size_t i = 0; i < losses.size(); ++i) {
		file << i + 1 << "," << losses[i] << "\n";
	}
	file.close();
	cout << "Loss log guardado en " << filename << endl;
}

// Guardar matriz de confusión
void save_confusion_matrix(const vector<vector<int>>& cm, const string& filename = "confusion_matrix.txt") {
	ofstream file(filename);
	if (!file.is_open()) {
		cerr << "Error al abrir " << filename << endl;
		return;
	}
	file << "True\\Pred,0,1,2,3,4,5,6,7,8,9\n";
	for (int i = 0; i < 10; ++i) {
		file << i;
		for (int j = 0; j < 10; ++j) {
			file << "," << cm[i][j];
		}
		file << "\n";
	}
	file.close();
	cout << "Matriz de confusión guardada en " << filename << endl;
}

int main() {
	// Entrenamiento con train set
	Data mnist_train(true); // true = train set
	auto inputs_train = mnist_train.get_normalized_images();
	auto outputs_train = mnist_train.get_one_hot_labels();

	cout << "TRAINING...\n";

	// Use the CUDA MLP
	MLP_CUDA* mlp_cuda = create_mlp_cuda({ 784, 5, 5, 10 });
	vector<float> epoch_losses; // Para almacenar el loss de cada época

	// Entrenamiento (modificado para guardar losses)
	int epochs = 10;
	float learning_rate = 0.1f;

	for (int epoch = 0; epoch < epochs; ++epoch) {
		float total_loss = 0.0f;
		for (size_t i = 0; i < inputs_train.size(); ++i) {
			total_loss += train_sample_cuda(*mlp_cuda, inputs_train[i], outputs_train[i], learning_rate);
		}
		float avg_loss = total_loss / inputs_train.size();
		epoch_losses.push_back(avg_loss);
		cout << "Epoch " << epoch + 1 << " completada - Loss promedio: " << avg_loss << "\n";
	}
	save_loss_log(epoch_losses); // Guarda el log de losses

	// Evaluación con test set
	Data mnist_test(false); // false = test set
	auto inputs_test = mnist_test.get_normalized_images();
	auto outputs_test = mnist_test.get_one_hot_labels();

	int correct = 0;
	vector<vector<int>> confusion_matrix(10, vector<int>(10, 0)); // Matriz 10x10 inicializada en 0

	for (size_t i = 0; i < inputs_test.size(); ++i) {
		auto prediction = predict_mlp_cuda(*mlp_cuda, inputs_test[i]);
		int pred_class = max_element(prediction.begin(), prediction.end()) - prediction.begin();
		int true_class = max_element(outputs_test[i].begin(), outputs_test[i].end()) - outputs_test[i].begin();

		confusion_matrix[true_class][pred_class]++; // Actualiza la matriz de confusión

		if (pred_class == true_class) correct++;
	}

	float accuracy = (correct * 100.0f) / inputs_test.size();
	cout << "Accuracy en test set: " << accuracy << "%" << endl;

	save_confusion_matrix(confusion_matrix); // Guarda la matriz de confusión

	destroy_mlp_cuda(mlp_cuda); // Free CUDA memory

	return 0;
}