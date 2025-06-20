// cuda_mlp.cuh
#ifndef CUDA_MLP_H
#define CUDA_MLP_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>  // Necesario para blockIdx, etc.
#include <vector>

struct Neuron;

// Estructura para capas en GPU
struct GPULayer {
    float* weights;
    float* biases;
    int num_neurons;
    int input_size;
};

// Kernels CUDA
__global__ void weightedSumKernel(const float* input, const float* weights,
    const float* biases, float* output,
    int num_neurons, int input_size);

__global__ void outputDeltaKernel(const float* activations, const float* targets,
    const float* zs, float* deltas,
    float* loss, int output_size);

// Funciones de gestión GPU
void copy_mlp_to_gpu(const std::vector<std::vector<float>>& weights,
    const std::vector<float>& biases,
    GPULayer* gpu_layer);

void free_gpu_layer(GPULayer* gpu_layer);

#endif