// cuda_mlp.cu
#include "cuda_mlp.cuh"
#include <iostream>

__global__ void outputDeltaKernel(const float* activations, const float* targets,
    const float* zs, float* deltas,
    float* loss, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        float error = activations[idx] - targets[idx];
        atomicAdd(loss, error * error);
        float sig_z = 1.0f / (1.0f + expf(-zs[idx]));
        deltas[idx] = error * sig_z * (1 - sig_z);
    }
}

// Kernel para forward pass
__global__ void weightedSumKernel(const float* input, const float* weights,
    const float* biases, float* output,
    int num_neurons, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_neurons) {
        float sum = biases[idx];
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weights[idx * input_size + i];
        }
        output[idx] = 1.0f / (1.0f + expf(-sum)); // Sigmoid
    }
}

// Copiar datos a GPU
void copy_mlp_to_gpu(const std::vector<std::vector<float>>& weights,
    const std::vector<float>& biases,
    GPULayer* gpu_layer) {
    int input_size = weights[0].size();
    int num_neurons = weights.size();

    // Flatten weights
    std::vector<float> flat_weights;
    for (const auto& w : weights) {
        flat_weights.insert(flat_weights.end(), w.begin(), w.end());
    }

    // Alloc memoria GPU
    cudaMalloc(&gpu_layer->weights, flat_weights.size() * sizeof(float));
    cudaMalloc(&gpu_layer->biases, biases.size() * sizeof(float));

    // Copiar datos
    cudaMemcpy(gpu_layer->weights, flat_weights.data(),
        flat_weights.size() * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_layer->biases, biases.data(),
        biases.size() * sizeof(float),
        cudaMemcpyHostToDevice);

    gpu_layer->input_size = input_size;
    gpu_layer->num_neurons = num_neurons;
}

// Liberar memoria GPU
void free_gpu_layer(GPULayer* gpu_layer) {
    cudaFree(gpu_layer->weights);
    cudaFree(gpu_layer->biases);
}