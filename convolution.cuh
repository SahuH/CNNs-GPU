#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <cstddef>


typedef struct {
    int width;
    int height;
    int depth;
    float *data;
} img;

typedef struct {
    int size; // length & width of kernel
    int depth;
    int n_filters;
    float *data;
} ker;

__host__ void convolution(img *input, img *output, ker *kernel, int stride);

__host__ void add_padding(img *input, img *output, int padding);

__host__ void pooling(img *input, img *output, int size, int stride);

__host__ void relu(img *input, img *output);

__host__ void dense(img *input, img *output, float* weights, float* biases, int n_neurons);

__host__ void convolution_cpu(img *input, img *output, ker *kernel, int stride);

__host__ void add_padding_cpu(img *input, img *output, int padding);

__host__ void pooling_cpu(img *input, img *output, int size, int stride);

__host__ void relu_cpu(img *input, img *output);

__host__ void print_img(img *input);

__host__ void print_ker(ker *kernel);

__host__ void dense_cpu(img *input, img *output, float* weights, float* biases, int n_neurons);

#endif