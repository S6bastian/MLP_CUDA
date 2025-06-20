// main.cpp
#include "MNIST_Reader.h"
#include "Data.h"
#include "cuda_mlp.cuh"
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


float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}

float sigmoid_derivative(float x) {
	float f = sigmoid(x);
	return f * (1 - f);
}


struct Neuron {
	vector<float> weights;
	float bias;

	Neuron(int input_size) {
		weights.resize(input_size);

		std::random_device rd;  // Entropía del sistema
		std::mt19937 gen(rd()); // Motor Mersenne Twister con semilla de rd
		std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

		for (float& w : weights) {
			w = dist(gen);
		}
		bias = dist(gen);
	}

	float activate(const vector<float>& input) {
		float sum = bias;
		for (size_t i = 0; i < input.size(); ++i)
			sum += input[i] * weights[i];
		return sigmoid(sum);
	}
};

struct Layer {
	std::vector<Neuron> neurons;
	Layer(int input_size, int num_neurons) {
		for (int i = 0; i < num_neurons; ++i)
			neurons.emplace_back(input_size);
	}

	vector<float> forward(const vector<float>& input) {
		vector<float> output;
		for (auto& neuron : neurons)
			output.push_back(neuron.activate(input));
		return output;
	}
};

struct MLP {
	std::vector<Layer> layers;
	std::vector<GPULayer> gpu_layers;

	MLP(const vector<int>& layer_sizes) {
		for (size_t i = 1; i < layer_sizes.size(); ++i)
			layers.emplace_back(layer_sizes[i - 1], layer_sizes[i]);
	}

	vector<float> predict(const vector<float>& input) {
		vector<float> out = input;
		for (auto& layer : layers)
			out = layer.forward(out);
		return out;
	}

	void train(const vector<vector<float>>& inputs,
		const vector<vector<float>>& targets,
		int epochs, float learning_rate) {
		for (int epoch = 0; epoch < epochs; ++epoch) {
			float total_loss = 0.0f;
			for (size_t i = 0; i < inputs.size(); ++i) {
				total_loss += train_sample(inputs[i], targets[i], learning_rate);
			}
			cout << "Epoch " << epoch + 1 << " completada - Loss promedio: "
				<< total_loss / inputs.size() << "\n";
		}
	}

	// Entrena un solo ejemplo y devuelve el loss (error cuadrático medio para ese ejemplo)
	float train_sample(const vector<float>& input,
		const vector<float>& target,
		float learning_rate) {
		vector<vector<float>> activations;
		vector<vector<float>> zs;

		vector<float> current = input;
		activations.push_back(current);
		for (auto& layer : layers) {
			vector<float> z, a;
			for (auto& neuron : layer.neurons) {
				float sum = neuron.bias;
				for (size_t i = 0; i < current.size(); ++i)
					sum += current[i] * neuron.weights[i];  		// <-------------------- PARALELIZAR es la suma de todos los (W*X)
				z.push_back(sum);
				a.push_back(sigmoid(sum));
			}
			zs.push_back(z);
			activations.push_back(a);
			current = a;
		}

		// Inicialización de deltas
		vector<vector<float>> deltas(layers.size());
		int L = layers.size() - 1;
		deltas[L].resize(layers[L].neurons.size());

		// Cálculo del error en la capa de salida
		float sample_loss = 0.0f;
		for (int i = 0; i < static_cast<int>(deltas[L].size()); ++i) {			// <-------------------- PARALELIZAR esto tambien se podría creo
			float error = activations.back()[i] - target[i];
			sample_loss += error * error;
			deltas[L][i] = error * sigmoid_derivative(zs[L][i]);
		}

		// Backpropagation hacia capas anteriores
		for (int l = L - 1; l >= 0; --l) {
			deltas[l].resize(layers[l].neurons.size());
			for (size_t i = 0; i < layers[l].neurons.size(); ++i) {
				float error = 0.0f;
				for (size_t j = 0; j < layers[l + 1].neurons.size(); ++j)
					error += deltas[l + 1][j] * layers[l + 1].neurons[j].weights[i]; 	// <-------------------- PARALELIZAR esto tambien se podría creo
				deltas[l][i] = error * sigmoid_derivative(zs[l][i]);
			}
		}

		// Actualización de pesos y biases
		for (size_t l = 0; l < layers.size(); ++l) {
			for (size_t i = 0; i < layers[l].neurons.size(); ++i) {
				for (size_t j = 0; j < layers[l].neurons[i].weights.size(); ++j)	// <-------------------- PARALELIZAR esto tambien se podría creo
					layers[l].neurons[i].weights[j] -= learning_rate * deltas[l][i] * activations[l][j];
				layers[l].neurons[i].bias -= learning_rate * deltas[l][i];
			}
		}

		// Retorno del loss
		return sample_loss / 2.0f; // MSE: (1/2) * error^2
	}

};

// Función para compatibilidad con Windows
#if defined(_WIN32) || defined(_WIN64)
#define bswap_32(x) _byteswap_ulong(x)
#else
#define bswap_32(x) __builtin_bswap32(x)
#endif

int main() {
	// Entrenamiento con train set
	Data mnist_train(true); // true = train set
	auto inputs_train = mnist_train.get_normalized_images();
	auto outputs_train = mnist_train.get_one_hot_labels();

	cout << "TRAINING...\n";

	MLP mlp({ 784, 4, 4, 10 });
	vector<float> epoch_losses; // Para almacenar el loss de cada época

	// Entrenamiento (modificado para guardar losses)
	int epochs = 1000;
	for (int epoch = 0; epoch < epochs; ++epoch) {
		float total_loss = 0.0f;
		for (size_t i = 0; i < inputs_train.size(); ++i) {
			total_loss += mlp.train_sample(inputs_train[i], outputs_train[i], 0.1f);
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
		auto prediction = mlp.predict(inputs_test[i]);
		int pred_class = max_element(prediction.begin(), prediction.end()) - prediction.begin();
		int true_class = max_element(outputs_test[i].begin(), outputs_test[i].end()) - outputs_test[i].begin();

		confusion_matrix[true_class][pred_class]++; // Actualiza la matriz de confusión

		if (pred_class == true_class) correct++;
	}

	float accuracy = (correct * 100.0f) / inputs_test.size();
	cout << "Accuracy en test set: " << accuracy << "%" << endl;

	save_confusion_matrix(confusion_matrix); // Guarda la matriz de confusión

	return 0;
}
