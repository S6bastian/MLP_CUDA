#include "mlp_cuda_types.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <algorithm>

// Check for CUDA errors
#define CHECK_CUDA_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Device-side sigmoid function
__device__ float sigmoid_dev(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Device-side sigmoid derivative function
__device__ float sigmoid_derivative_dev(float x) {
    float f = sigmoid_dev(x);
    return f * (1 - f);
}


// Structure for a Neuron on the device
struct Neuron_dev {
    float* weights;
    float bias;
    int input_size;
};

// Structure for a Layer on the device
struct Layer_dev {
    Neuron_dev* neurons;
    int num_neurons;
    int input_size; // input size for this layer
};

// Structure for the MLP on the device
struct MLP_dev {
    Layer_dev* layers;
    int num_layers;
};

// --- KERNELS ---

// Kernel for forward pass of a layer
__global__ void forward_layer_kernel(const float* input, int input_size,
    const Neuron_dev* neurons, int num_neurons,
    float* output_activations, float* output_zs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_neurons) {
        float sum = neurons[idx].bias;
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * neurons[idx].weights[i];
        }
        output_zs[idx] = sum;
        output_activations[idx] = sigmoid_dev(sum);
    }
}

// Kernel for calculating output layer deltas
__global__ void calculate_output_deltas_kernel(const float* activations_last_layer,
    const float* target,
    const float* zs_last_layer,
    int num_neurons_last_layer,
    float* deltas_last_layer,
    float* sample_loss_ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_neurons_last_layer) {
        float error = activations_last_layer[idx] - target[idx];
        atomicAdd(sample_loss_ptr, error * error); // Accumulate loss atomically
        deltas_last_layer[idx] = error * sigmoid_derivative_dev(zs_last_layer[idx]);
    }
}

// Kernel for backpropagating deltas to previous layers
__global__ void backpropagate_deltas_kernel(const float* next_layer_deltas, int next_layer_num_neurons,
    const Neuron_dev* next_layer_neurons, // to access weights
    const float* current_layer_zs,
    int current_layer_num_neurons,
    float* current_layer_deltas) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < current_layer_num_neurons) {
        float error = 0.0f;
        for (int j = 0; j < next_layer_num_neurons; ++j) {
            error += next_layer_deltas[j] * next_layer_neurons[j].weights[idx];
        }
        current_layer_deltas[idx] = error * sigmoid_derivative_dev(current_layer_zs[idx]);
    }
}


// Kernel for updating weights and biases
__global__ void update_weights_biases_kernel(float* weights, float* bias,
    const float* deltas_current_layer,
    const float* activations_prev_layer,
    int num_neurons_current_layer,
    int prev_layer_num_neurons, // input size for current layer
    float learning_rate) {
    int idx_neuron = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_neuron < num_neurons_current_layer) {
        // Update bias
        bias[idx_neuron] -= learning_rate * deltas_current_layer[idx_neuron];

        // Update weights
        for (int j = 0; j < prev_layer_num_neurons; ++j) {
            weights[idx_neuron * prev_layer_num_neurons + j] -= learning_rate * deltas_current_layer[idx_neuron] * activations_prev_layer[j];
        }
    }
}

// --- HOST-SIDE STRUCTS ---

// Host-side representation of Neuron
struct Neuron {
    std::vector<float> weights;
    float bias;

    Neuron(int input_size) : weights(input_size), bias(0.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (float& w : weights) {
            w = dist(gen);
        }
        bias = dist(gen);
    }
};

// Host-side representation of Layer
struct Layer {
    std::vector<Neuron> neurons;
    int input_size;
    int num_neurons;

    Layer(int input_size, int num_neurons) : input_size(input_size), num_neurons(num_neurons) {
        for (int i = 0; i < num_neurons; ++i) {
            neurons.emplace_back(input_size);
        }
    }
};

// Host-side representation of MLP, managing device memory
struct MLP_CUDA {
    std::vector<Layer> host_layers; // Original host layers for initialization and final retrieval
    MLP_dev* d_mlp;                 // Device MLP structure pointer on host
    MLP_dev h_mlp_config;           // Host copy of MLP_dev structure

    // Pointers to device memory for activations, zs, and deltas for each layer
    std::vector<float*> d_activations;
    std::vector<float*> d_zs;
    std::vector<float*> d_deltas;

    // Pointers to device memory for weights and biases of each layer
    std::vector<float*> d_weights_ptrs;
    std::vector<float*> d_biases_ptrs;

    MLP_CUDA(const std::vector<int>& layer_sizes) {
        if (layer_sizes.size() < 2) {
            std::cerr << "MLP must have at least an input and an output layer." << std::endl;
            exit(1);
        }

        // Initialize host layers
        for (size_t i = 1; i < layer_sizes.size(); ++i) {
            host_layers.emplace_back(layer_sizes[i - 1], layer_sizes[i]);
        }

        int num_layers = host_layers.size();
        h_mlp_config.num_layers = num_layers; // Set num_layers in host-side MLP_dev

        // Allocate memory for the array of Layer_dev structures on the device
        CHECK_CUDA_ERROR(cudaMalloc(&h_mlp_config.layers, num_layers * sizeof(Layer_dev)));

        // Allocate the main MLP_dev structure on the device and get its address
        CHECK_CUDA_ERROR(cudaMalloc(&d_mlp, sizeof(MLP_dev)));

        d_activations.resize(num_layers + 1); // input layer + hidden layers + output layer
        d_zs.resize(num_layers);              // zs for hidden and output layers
        d_deltas.resize(num_layers);          // deltas for hidden and output layers
        d_weights_ptrs.resize(num_layers);
        d_biases_ptrs.resize(num_layers);

        std::vector<Layer_dev> h_layer_dev_configs(num_layers); // Temporary host storage for device layer configs

        // Allocate and copy data for each layer
        for (int l = 0; l < num_layers; ++l) {
            Layer& current_host_layer = host_layers[l];
            int input_size_for_layer = current_host_layer.input_size;
            int num_neurons_in_layer = current_host_layer.num_neurons;

            // Setup current layer_dev config on host
            h_layer_dev_configs[l].num_neurons = num_neurons_in_layer;
            h_layer_dev_configs[l].input_size = input_size_for_layer;

            // Allocate memory for neurons in the device layer
            CHECK_CUDA_ERROR(cudaMalloc(&h_layer_dev_configs[l].neurons, num_neurons_in_layer * sizeof(Neuron_dev)));

            // Allocate and copy weights and biases for each neuron
            float* d_layer_weights_flat;
            float* d_layer_biases;
            size_t total_weights_size = 0;
            for (const auto& neuron : current_host_layer.neurons) {
                total_weights_size += neuron.weights.size();
            }

            CHECK_CUDA_ERROR(cudaMalloc(&d_layer_weights_flat, total_weights_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_layer_biases, num_neurons_in_layer * sizeof(float)));

            d_weights_ptrs[l] = d_layer_weights_flat; // Store pointer to weights for later access
            d_biases_ptrs[l] = d_layer_biases;       // Store pointer to biases for later access

            std::vector<Neuron_dev> temp_neuron_devs(num_neurons_in_layer);
            size_t current_weight_offset = 0;
            for (int n = 0; n < num_neurons_in_layer; ++n) {
                Neuron& host_neuron = current_host_layer.neurons[n];

                // Copy weights to the flattened device array
                CHECK_CUDA_ERROR(cudaMemcpy(d_layer_weights_flat + current_weight_offset, host_neuron.weights.data(),
                    host_neuron.weights.size() * sizeof(float), cudaMemcpyHostToDevice));

                temp_neuron_devs[n].weights = d_layer_weights_flat + current_weight_offset;
                temp_neuron_devs[n].bias = host_neuron.bias;
                temp_neuron_devs[n].input_size = host_neuron.weights.size();
                current_weight_offset += host_neuron.weights.size();

                CHECK_CUDA_ERROR(cudaMemcpy(d_layer_biases + n, &host_neuron.bias, sizeof(float), cudaMemcpyHostToDevice));
            }
            CHECK_CUDA_ERROR(cudaMemcpy(h_layer_dev_configs[l].neurons, temp_neuron_devs.data(), num_neurons_in_layer * sizeof(Neuron_dev), cudaMemcpyHostToDevice));

            // Allocate memory for activations, zs, deltas
            int current_layer_output_size = current_host_layer.num_neurons;
            CHECK_CUDA_ERROR(cudaMalloc(&d_activations[l + 1], current_layer_output_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_zs[l], current_layer_output_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_deltas[l], current_layer_output_size * sizeof(float)));
        }
        // Allocate for the input activation (d_activations[0])
        CHECK_CUDA_ERROR(cudaMalloc(&d_activations[0], layer_sizes[0] * sizeof(float)));

        // Copy the array of Layer_dev structures from host to device
        CHECK_CUDA_ERROR(cudaMemcpy(h_mlp_config.layers, h_layer_dev_configs.data(), num_layers * sizeof(Layer_dev), cudaMemcpyHostToDevice));

        // Copy the final MLP_dev structure (containing pointers to device memory) to the device
        CHECK_CUDA_ERROR(cudaMemcpy(d_mlp, &h_mlp_config, sizeof(MLP_dev), cudaMemcpyHostToDevice));
    }

    // Destructor to free device memory
    ~MLP_CUDA() {
        int num_layers = host_layers.size();
        // First, copy the MLP_dev structure back to host to access its pointers
        MLP_dev temp_d_mlp_config;
        CHECK_CUDA_ERROR(cudaMemcpy(&temp_d_mlp_config, d_mlp, sizeof(MLP_dev), cudaMemcpyDeviceToHost));

        std::vector<Layer_dev> temp_h_layer_dev_configs(num_layers);
        CHECK_CUDA_ERROR(cudaMemcpy(temp_h_layer_dev_configs.data(), temp_d_mlp_config.layers, num_layers * sizeof(Layer_dev), cudaMemcpyDeviceToHost));


        for (int l = 0; l < num_layers; ++l) {
            CHECK_CUDA_ERROR(cudaFree(temp_h_layer_dev_configs[l].neurons)); // Free array of Neuron_dev

            CHECK_CUDA_ERROR(cudaFree(d_weights_ptrs[l]));
            CHECK_CUDA_ERROR(cudaFree(d_biases_ptrs[l]));

            CHECK_CUDA_ERROR(cudaFree(d_activations[l + 1]));
            CHECK_CUDA_ERROR(cudaFree(d_zs[l]));
            CHECK_CUDA_ERROR(cudaFree(d_deltas[l]));
        }
        CHECK_CUDA_ERROR(cudaFree(d_activations[0]));
        CHECK_CUDA_ERROR(cudaFree(temp_d_mlp_config.layers)); // Free array of Layer_dev
        CHECK_CUDA_ERROR(cudaFree(d_mlp)); // Free the main MLP_dev structure
    }
};

MLP_CUDA* create_mlp_cuda(const std::vector<int>& layer_sizes) {
    return new MLP_CUDA(layer_sizes);
}

void destroy_mlp_cuda(MLP_CUDA* mlp) {
    delete mlp;
}

// Function to perform a single training sample on the GPU
float train_sample_cuda(MLP_CUDA& mlp, const std::vector<float>& input,
    const std::vector<float>& target, float learning_rate) {
    int num_layers = mlp.host_layers.size(); // Number of weight layers

    // 1. Forward Pass
    float* d_current_input = mlp.d_activations[0]; // Start with the input layer's activations
    CHECK_CUDA_ERROR(cudaMemcpy(d_current_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    for (int l = 0; l < num_layers; ++l) {
        // Need to get the Layer_dev struct for the current layer from the device
        // This is inefficient. Ideally, the kernel should take a pointer to the array of Layer_dev on the device.
        // For now, let's keep the current approach, but know this is a point for optimization.
        // The d_mlp->layers in the d_mlp structure on device *already* points to the array of Layer_dev on device.
        // We just need to ensure the kernels have access to this.

        Layer_dev d_current_layer_dev_struct;
        // Copy one Layer_dev structure from the device array to a host temporary
        CHECK_CUDA_ERROR(cudaMemcpy(&d_current_layer_dev_struct, mlp.h_mlp_config.layers + l, sizeof(Layer_dev), cudaMemcpyDeviceToHost));

        float* d_output_activations = mlp.d_activations[l + 1];
        float* d_output_zs = mlp.d_zs[l];

        int num_neurons = mlp.host_layers[l].num_neurons;
        int input_size_for_layer = mlp.host_layers[l].input_size;

        int threadsPerBlock = 256;
        int numBlocks = (num_neurons + threadsPerBlock - 1) / threadsPerBlock;

        forward_layer_kernel << <numBlocks, threadsPerBlock >> > (
            d_current_input, input_size_for_layer,
            d_current_layer_dev_struct.neurons, num_neurons, // Pass the device pointer to neurons within this layer
            d_output_activations, d_output_zs
            );
        CHECK_CUDA_ERROR(cudaGetLastError());
        d_current_input = d_output_activations;
    }

    // 2. Backward Pass - Calculate output layer deltas and loss
    float* d_target;
    CHECK_CUDA_ERROR(cudaMalloc(&d_target, target.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_target, target.data(), target.size() * sizeof(float), cudaMemcpyHostToDevice));

    float* d_sample_loss;
    CHECK_CUDA_ERROR(cudaMalloc(&d_sample_loss, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_sample_loss, 0, sizeof(float)));

    int L = num_layers - 1;
    int output_layer_num_neurons = mlp.host_layers[L].num_neurons;

    int threadsPerBlock = 256;
    int numBlocks = (output_layer_num_neurons + threadsPerBlock - 1) / threadsPerBlock;

    calculate_output_deltas_kernel << <numBlocks, threadsPerBlock >> > (
        mlp.d_activations[L + 1], d_target, mlp.d_zs[L],
        output_layer_num_neurons, mlp.d_deltas[L], d_sample_loss
        );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    float h_sample_loss;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_sample_loss, d_sample_loss, sizeof(float), cudaMemcpyDeviceToHost));
    h_sample_loss /= 2.0f;

    CHECK_CUDA_ERROR(cudaFree(d_target));
    CHECK_CUDA_ERROR(cudaFree(d_sample_loss));


    // 3. Backpropagate deltas to previous layers
    for (int l = L - 1; l >= 0; --l) {
        int current_layer_num_neurons = mlp.host_layers[l].num_neurons;

        Layer_dev d_next_layer_dev_struct;
        CHECK_CUDA_ERROR(cudaMemcpy(&d_next_layer_dev_struct, mlp.h_mlp_config.layers + (l + 1), sizeof(Layer_dev), cudaMemcpyDeviceToHost));

        threadsPerBlock = 256;
        numBlocks = (current_layer_num_neurons + threadsPerBlock - 1) / threadsPerBlock;

        backpropagate_deltas_kernel << <numBlocks, threadsPerBlock >> > (
            mlp.d_deltas[l + 1], mlp.host_layers[l + 1].num_neurons,
            d_next_layer_dev_struct.neurons,
            mlp.d_zs[l], current_layer_num_neurons,
            mlp.d_deltas[l]
            );
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    // 4. Update weights and biases
    for (int l = 0; l < num_layers; ++l) {
        int current_layer_num_neurons = mlp.host_layers[l].num_neurons;
        int prev_layer_num_neurons = mlp.host_layers[l].input_size;

        threadsPerBlock = 256;
        numBlocks = (current_layer_num_neurons + threadsPerBlock - 1) / threadsPerBlock;

        update_weights_biases_kernel << <numBlocks, threadsPerBlock >> > (
            mlp.d_weights_ptrs[l], mlp.d_biases_ptrs[l],
            mlp.d_deltas[l], mlp.d_activations[l],
            current_layer_num_neurons, prev_layer_num_neurons,
            learning_rate
            );
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy updated weights/biases back to host for MLP_CUDA internal state
    // This is still done per sample, which is a major bottleneck.
    // For large datasets, consider copying back only once per epoch or at the end of training.
    for (int l = 0; l < num_layers; ++l) {
        Layer& host_layer = mlp.host_layers[l];
        int num_neurons_in_layer = host_layer.num_neurons;
        int input_size_for_layer = host_layer.input_size;

        float* h_layer_biases = new float[num_neurons_in_layer];
        CHECK_CUDA_ERROR(cudaMemcpy(h_layer_biases, mlp.d_biases_ptrs[l], num_neurons_in_layer * sizeof(float), cudaMemcpyDeviceToHost));

        size_t total_weights_size = num_neurons_in_layer * input_size_for_layer;
        float* h_layer_weights_flat = new float[total_weights_size];
        CHECK_CUDA_ERROR(cudaMemcpy(h_layer_weights_flat, mlp.d_weights_ptrs[l], total_weights_size * sizeof(float), cudaMemcpyDeviceToHost));

        for (int n = 0; n < num_neurons_in_layer; ++n) {
            host_layer.neurons[n].bias = h_layer_biases[n];
            for (int w_idx = 0; w_idx < input_size_for_layer; ++w_idx) {
                host_layer.neurons[n].weights[w_idx] = h_layer_weights_flat[n * input_size_for_layer + w_idx];
            }
        }
        delete[] h_layer_biases;
        delete[] h_layer_weights_flat;
    }

    return h_sample_loss;
}

// Function to train the MLP using CUDA
void train_mlp_cuda(MLP_CUDA& mlp, const std::vector<std::vector<float>>& inputs,
    const std::vector<std::vector<float>>& targets,
    int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < inputs.size(); ++i) {
            total_loss += train_sample_cuda(mlp, inputs[i], targets[i], learning_rate);
        }
        std::cout << "Epoch " << epoch + 1 << " completada - Loss promedio: "
            << total_loss / inputs.size() << "\n";
    }
}

// Function to predict using the trained MLP on the GPU
std::vector<float> predict_mlp_cuda(MLP_CUDA& mlp, const std::vector<float>& input) {
    int num_layers = mlp.host_layers.size();

    float* d_current_input = mlp.d_activations[0];
    CHECK_CUDA_ERROR(cudaMemcpy(d_current_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    for (int l = 0; l < num_layers; ++l) {
        Layer_dev d_current_layer_dev_struct;
        CHECK_CUDA_ERROR(cudaMemcpy(&d_current_layer_dev_struct, mlp.h_mlp_config.layers + l, sizeof(Layer_dev), cudaMemcpyDeviceToHost));

        float* d_output_activations = mlp.d_activations[l + 1];
        float* d_output_zs = mlp.d_zs[l]; // Not strictly needed for prediction, but kept for consistency if using the same kernel

        int num_neurons = mlp.host_layers[l].num_neurons;
        int input_size_for_layer = mlp.host_layers[l].input_size;

        int threadsPerBlock = 256;
        int numBlocks = (num_neurons + threadsPerBlock - 1) / threadsPerBlock;

        forward_layer_kernel << <numBlocks, threadsPerBlock >> > (
            d_current_input, input_size_for_layer,
            d_current_layer_dev_struct.neurons, num_neurons,
            d_output_activations, d_output_zs
            );
        CHECK_CUDA_ERROR(cudaGetLastError());
        d_current_input = d_output_activations;
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    std::vector<float> prediction(mlp.host_layers.back().num_neurons);
    CHECK_CUDA_ERROR(cudaMemcpy(prediction.data(), mlp.d_activations[num_layers],
        prediction.size() * sizeof(float), cudaMemcpyDeviceToHost));

    return prediction;
}