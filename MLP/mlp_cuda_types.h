#ifndef MLP_CUDA_H
#define MLP_CUDA_H

#include <vector>

// Forward declarations of structs
struct Neuron;
struct Layer;
struct MLP_CUDA;

// Host-side wrapper functions
// Add the declaration for train_sample_cuda here
float train_sample_cuda(MLP_CUDA& mlp, const std::vector<float>& input,
    const std::vector<float>& target, float learning_rate);

void train_mlp_cuda(MLP_CUDA& mlp, const std::vector<std::vector<float>>& inputs,
    const std::vector<std::vector<float>>& targets,
    int epochs, float learning_rate);

std::vector<float> predict_mlp_cuda(MLP_CUDA& mlp, const std::vector<float>& input);

MLP_CUDA* create_mlp_cuda(const std::vector<int>& layer_sizes);

void destroy_mlp_cuda(MLP_CUDA* mlp);

#endif // MLP_CUDA_H