#include <stdio.h>
#include <iostream>
#include <random>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include "convolution.cuh"

__host__ img* testlayer(img *input, ker *kernel, int padding, int stride) {

    // padding
    img *padded_input;
    cudaMallocManaged((void**)&padded_input, sizeof(img));
    add_padding(input, padded_input, padding);

    // convolution
    img *conv_output;
    cudaMallocManaged((void**)&conv_output, sizeof(img));
    convolution(padded_input, conv_output, kernel, stride);

    // relu
    img *relu_output;
    cudaMallocManaged((void**)&relu_output, sizeof(img));
    relu(conv_output, relu_output);

    // pooling
    img *pool_output;
    cudaMallocManaged((void**)&pool_output, sizeof(img));
    pooling(relu_output, pool_output, 2, 2);

    // free memory
    cudaFree(padded_input->data);
    cudaFree(conv_output->data);
    cudaFree(relu_output->data);
    cudaFree(padded_input);
    cudaFree(conv_output);
    cudaFree(relu_output);

    return pool_output;
}

__host__ img* testlayer_cpu(img *input, ker *kernel, int padding, int stride) {

    // padding
    img *padded_input;
    cudaMallocManaged((void**)&padded_input, sizeof(img));
    add_padding_cpu(input, padded_input, padding);

    // convolution
    img *conv_output;
    cudaMallocManaged((void**)&conv_output, sizeof(img));
    convolution_cpu(padded_input, conv_output, kernel, stride);

    // relu
    img *relu_output;
    cudaMallocManaged((void**)&relu_output, sizeof(img));
    relu_cpu(conv_output, relu_output);

    // pooling
    img *pool_output;
    cudaMallocManaged((void**)&pool_output, sizeof(img));
    pooling_cpu(relu_output, pool_output, 2, 2);

    // free memory
    cudaFree(padded_input->data);
    cudaFree(conv_output->data);
    cudaFree(relu_output->data);
    cudaFree(padded_input);
    cudaFree(conv_output);
    cudaFree(relu_output);

    return pool_output;
}

int main(int argc, char** argv) {

        if(argc != 3){
        printf("Usage: ./convolution <input size> <kernel size>\n");
        }

        int input_size = atoi(argv[1]);
        int kernel_size = atoi(argv[2]);

        img *input;
        ker *kernel;

        cudaMallocManaged((void**)&input, sizeof(img));
        cudaMallocManaged((void**)&kernel, sizeof(ker));

        input->width = input_size;
        input->height = input_size;
        input->depth = 3;
        cudaMallocManaged(&input->data, sizeof(float) * input->width * input->height * input->depth);

        kernel->size = kernel_size;
        kernel->depth = input->depth;
        kernel->n_filters = 2;
        cudaMallocManaged(&kernel->data, sizeof(float) * kernel->size * kernel->size * kernel->depth * kernel->n_filters);

        int seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 generator(seed);

        const float _imgMin = -10.0f, _imgMax = 10.0f;
        std::uniform_real_distribution<float> imgDist(_imgMin, _imgMax);

        const float _kerMin = -1.0f, _kerMax = 1.0f;
        std::uniform_real_distribution<float> kerDist(_kerMin, _kerMax);

        for (int i = 0; i < input->width * input->height * input->depth; i++) {
        input->data[i] = imgDist(generator);
        }

        for(int i = 0; i<kernel->size * kernel->size * kernel->depth * kernel->n_filters; i++){
        kernel->data[i] = kerDist(generator);
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        img* output = testlayer(input, kernel, 1, 1);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("%f\n", milliseconds);

        //printf("output first element: %f\n", output->data[0]);
        //printf("output last element: %f\n", output->data[output->width * output->height * output->depth - 1]);

        cudaFree(output->data);
        cudaFree(output);

        // now test cpu

        cudaEventRecord(start);
        
        img* output_cpu = testlayer_cpu(input, kernel, 1, 1);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;

        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("%f\n", milliseconds);

        //printf("CPU output first element: %f\n", output_cpu->data[0]);
        //printf("CPU output last element: %f\n", output_cpu->data[output_cpu->width * output_cpu->height * output_cpu->depth - 1]);

        cudaFree(input->data);
        cudaFree(output->data);
        cudaFree(kernel->data);
        cudaFree(input);
        cudaFree(output);
        cudaFree(kernel);
        return 0;
}
