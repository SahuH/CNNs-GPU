#include <stdio.h>
#include <iostream>
#include <random>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include "convolution.cuh"

#define POOLING_SIZE 2
#define POOLING_STRIDE 2
#define CONV_STRIDE 1
#define RANDOM_MIN -1.0f
#define RANDOM_MAX 1.0f

__host__ void fill_random(float* data, int size, float min, float max) {
        int seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(min, max);
        for (int i = 0; i < size; i++) {
                data[i] = distribution(generator);
        }
}

__host__ img* initiate_input(int width, int height, int depth) {
        img* input;
        cudaMallocManaged((void**)&input, sizeof(img));
        input->width = width;
        input->height = height;
        input->depth = depth;
        cudaMallocManaged((void**)&input->data, sizeof(float) * input->width * input->height * input->depth);
        fill_random(input->data, input->width * input->height * input->depth, RANDOM_MIN, RANDOM_MAX);
        return input;
}

__host__ ker* initiate_kernel(int size, int depth, int n_filters) {
        ker* kernel;
        cudaMallocManaged((void**)&kernel, sizeof(ker));
        kernel->size = size;
        kernel->depth = depth;
        kernel->n_filters = n_filters;
        cudaMallocManaged((void**)&kernel->data, sizeof(float) * kernel->size * kernel->size * kernel->depth * kernel->n_filters);
        fill_random(kernel->data, kernel->size * kernel->size * kernel->depth * kernel->n_filters, RANDOM_MIN, RANDOM_MAX);
        return kernel;
}

__host__ float* initiate_dense_layer_weight(int n_inputs, int n_outputs) {
        float* weights;
        cudaMallocManaged((void**)&weights, sizeof(float) * n_inputs * n_outputs);
        fill_random(weights, n_inputs * n_outputs, -1.0f, 1.0f);
        return weights;
}

__host__ float* initiate_dense_layer_bias(int n_outputs) {
        float* biases;
        cudaMallocManaged((void**)&biases, sizeof(float) * n_outputs);
        fill_random(biases, n_outputs, -1.0f, 1.0f);
        return biases;
}

int main(int argc, char** argv) {

    if(argc != 8) {
        std::cout << "Usage: ./benchmarkModels <input_size> <kernel_size> <n_filters> <n_conv_layers> <n_pooling_layers> <n_dense_layers> <n_neurons>" << std::endl;
        return 1;
    }

    int input_size = atoi(argv[1]);

    int kernel_size = atoi(argv[2]);

    int filter_size = atoi(argv[3]);

    int n_conv_layers = atoi(argv[4]);

    int n_pooling_layers = atoi(argv[5]);

    int n_dense_layers = atoi(argv[6]);

    int n_neurons = atoi(argv[7]);

    img* input = NULL;

    ker** kernels;

    float** dense_layers_weights;
    float** dense_layers_biases;

    // printf("Initiating input\n");

    input = initiate_input(input_size, input_size, 3);

    // printf("Initiated input\n");

    kernels = (ker**)malloc(sizeof(ker*) * n_conv_layers);

    // printf("Initiating kernels\n");

    for(int i = 0; i < n_conv_layers; i++) {
        if(i == 0) {
            kernels[i] = initiate_kernel(kernel_size, input->depth, filter_size);
        }
        else {
            kernels[i] = initiate_kernel(kernel_size, kernels[i - 1]->n_filters, filter_size);
        }
    }

    // printf("Initiated kernels\n");

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    img* output;
    cudaMallocManaged((void**)&output, sizeof(img));

    for(int i = 0; i < n_conv_layers; i++) {

        // printf("Convolution layer %d\n", i);

        int padding = (kernels[i]->size - 1) / 2;
        add_padding_cpu(input, output, padding);
        cudaFree(input);
        input = output;

        // printf("Padding done\n");

        convolution_cpu(input, output, kernels[i], CONV_STRIDE);
        cudaFree(input);
        input = output;

        // printf("Convolution done\n");

        relu_cpu(input, output);
        cudaFree(input);
        input = output;

        // printf("Relu done\n");
    }

    for(int i = 0; i < n_pooling_layers; i++) {
        pooling_cpu(input, output, POOLING_SIZE, POOLING_STRIDE);
        cudaFree(input);
        input = output;

        // printf("Pooling layer %d\n", i);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    for(int i = 0; i < n_conv_layers; i++) {
        cudaFree(kernels[i]->data);
        cudaFree(kernels[i]);
    }

    cudaFree(kernels);

    dense_layers_weights = (float**)malloc(sizeof(float*) * n_dense_layers);
    dense_layers_biases = (float**)malloc(sizeof(float*) * n_dense_layers);

    for(int i = 0; i < n_dense_layers; i++) {
        if(i == 0) {
            dense_layers_weights[i] = initiate_dense_layer_weight(input->width * input->height * input->depth, n_neurons);
            dense_layers_biases[i] = initiate_dense_layer_bias(n_neurons);
        }
        else {
            dense_layers_weights[i] = initiate_dense_layer_weight(n_neurons, n_neurons);
            dense_layers_biases[i] = initiate_dense_layer_bias(n_neurons);
        }
    }

    // printf("Initiated dense layers\n");

    cudaEventRecord(start);

    img* dense_layer_output;
    cudaMallocManaged((void**)&dense_layer_output, sizeof(float) * n_neurons);
    for(int i = 0; i < n_dense_layers; i++) {
        dense_cpu(input, dense_layer_output, dense_layers_weights[i], dense_layers_biases[i], n_neurons);
        cudaFree(input);
        input = dense_layer_output;

        // printf("Dense layer %d\n", i);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // printf("Dense layers done\n");

    float millisecondsDense = 0;
    cudaEventElapsedTime(&millisecondsDense, start, stop);

    float totalMilliseconds = milliseconds + millisecondsDense;

    printf("%f\n", totalMilliseconds);

    cudaFree(input->data);
    cudaFree(input);
    
    cudaFree(output->data);
    cudaFree(output);

    for(int i = 0; i < n_dense_layers; i++) {
        cudaFree(dense_layers_weights[i]);
        cudaFree(dense_layers_biases[i]);
    }

    cudaFree(dense_layer_output);
    cudaFree(dense_layer_output->data);
    cudaFree(dense_layers_weights);
    cudaFree(dense_layers_biases);

}