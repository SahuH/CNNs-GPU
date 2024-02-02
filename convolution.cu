#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <random>
#include <chrono>

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

__global__ void convolution_kernel(img *input, img *output, ker *kernel, int stride) {
    extern __shared__ float shared_mem[];

    int shared_mem_size = kernel->size * kernel->size * kernel->depth * kernel->n_filters;
    for (int i = 0; i < shared_mem_size; i++) {
        shared_mem[i] = kernel->data[i];
    }

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < output->width && y < output->height && z < output->depth) {
        float sum = 0.0f;
        for (int i = 0; i < kernel->size; i++) {
            for (int j = 0; j < kernel->size; j++) {
                for (int k = 0; k < kernel->depth; k++) {
                    int input_x = x * stride + i;
                    int input_y = y * stride + j;
                    int input_z = k;
                    if (input_x >= 0 && input_x < input->width && input_y >= 0 && input_y < input->height) {
                        sum += input->data[input_z * input->width * input->height + input_y * input->width + input_x] *
                               shared_mem[kernel->size * kernel->size * kernel->depth * z + kernel->size * kernel->size * k + kernel->size * j + i];
                    }
                }
            }
        }
        output->data[z * output->width * output->height + y * output->width + x] = sum;
    }
}

__host__ void convolution(img *input, img *output, ker *kernel, int stride) {

    output->width = (input->width - kernel->size) / stride + 1;
    output->height = (input->height - kernel->size) / stride + 1;
    output->depth = kernel->n_filters;
    cudaMallocManaged(&output->data, sizeof(float) * output->width * output->height * output->depth);

    dim3 block_size(16, 16, 4);
    dim3 num_blocks(ceil(output->width / (float) block_size.x), ceil(output->height / (float) block_size.y),
                    ceil(output->depth / (float) block_size.z));

    int shared_mem_size = kernel->size * kernel->size * kernel->depth * kernel->n_filters * sizeof(float);

    convolution_kernel<<<num_blocks, block_size, shared_mem_size>>>(input, output, kernel, stride);
    cudaDeviceSynchronize();
}

__global__ void pooling_kernel(img *input, img *output, int size, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < output->width && y < output->height && z < output->depth) {
        float max = -INFINITY;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int input_x = x * stride + i;
                int input_y = y * stride + j;
                int input_z = z;
                if (input_x >= 0 && input_x < input->width && input_y >= 0 && input_y < input->height) {
                    float val = input->data[input_z * input->width * input->height + input_y * input->width + input_x];
                    if (val > max) {
                        max = val;
                    }
                }
            }
        }
        output->data[z * output->width * output->height + y * output->width + x] = max;
    }
}

__host__ void pooling(img *input, img *output, int size, int stride) {
    output->width = (input->width - size) / stride + 1;
    output->height = (input->height - size) / stride + 1;
    output->depth = input->depth;
    cudaMallocManaged(&output->data, sizeof(float) * output->width * output->height * output->depth);

    dim3 block_size(16, 16, 4);
    dim3 num_blocks(ceil(output->width / (float) block_size.x), ceil(output->height / (float) block_size.y),
                    ceil(output->depth / (float) block_size.z));

    pooling_kernel<<<num_blocks, block_size>>>(input, output, size, stride);
    cudaDeviceSynchronize();
}

__global__ void relu_kernel(img *input, img *output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < output->width && y < output->height && z < output->depth) {
        output->data[z * output->width * output->height + y * output->width + x] =
                input->data[z * input->width * input->height + y * input->width + x] > 0 ?
                input->data[z * input->width * input->height + y * input->width + x] : 0;
    }
}

__host__ void relu(img *input, img *output) {
    output->width = input->width;
    output->height = input->height;
    output->depth = input->depth;
    cudaMallocManaged(&output->data, sizeof(float) * output->width * output->height * output->depth);

    dim3 block_size(16, 16, 4);
    dim3 num_blocks(ceil(output->width / (float) block_size.x), ceil(output->height / (float) block_size.y),
                    ceil(output->depth / (float) block_size.z));

    relu_kernel<<<num_blocks, block_size>>>(input, output);
    cudaDeviceSynchronize();
}

__host__ void pooling_cpu(img *input, img *output, int size, int stride) {
    output->width = (input->width - size) / stride + 1;
    output->height = (input->height - size) / stride + 1;
    output->depth = input->depth;
    cudaMallocManaged(&output->data, sizeof(float) * output->width * output->height * output->depth);

    for (int x = 0; x < output->width; x++) {
        for (int y = 0; y < output->height; y++) {
            for (int z = 0; z < output->depth; z++) {
                float max = -INFINITY;
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        int input_x = x * stride + i;
                        int input_y = y * stride + j;
                        int input_z = z;
                        if (input_x >= 0 && input_x < input->width && input_y >= 0 && input_y < input->height) {
                            float val = input->data[input_z * input->width * input->height + input_y * input->width +
                                                    input_x];
                            if (val > max) {
                                max = val;
                            }
                        }
                    }
                }
                output->data[z * output->width * output->height + y * output->width + x] = max;
            }
        }
    }
}

__host__ void convolution_cpu(img *input, img *output, ker *kernel, int stride) {
    output->width = (input->width - kernel->size) / stride + 1;
    output->height = (input->height - kernel->size) / stride + 1;
    output->depth = kernel->n_filters;
    cudaMallocManaged(&output->data, sizeof(float) * output->width * output->height * output->depth);

    for (int x = 0; x < output->width; x++) {
        for (int y = 0; y < output->height; y++) {
            for (int z = 0; z < output->depth; z++) {
                float sum = 0.0f;
                for (int i = 0; i < kernel->size; i++) {
                    for (int j = 0; j < kernel->size; j++) {
                        for (int k = 0; k < kernel->depth; k++) {
                            int input_x = x * stride + i;
                            int input_y = y * stride + j;
                            int input_z = k;
                            if (input_x >= 0 && input_x < input->width && input_y >= 0 && input_y < input->height) {
                                sum += input->data[input_z * input->width * input->height + input_y * input->width +
                                                   input_x] *
                                       kernel->data[kernel->size * kernel->size * kernel->depth * z +
                                                   kernel->size * kernel->size * k + kernel->size * j + i];
                            }
                        }
                    }
                }
                output->data[z * output->width * output->height + y * output->width + x] = sum;
            }
        }
    }
}

__host__ void relu_cpu(img *input, img *output) {
    output->width = input->width;
    output->height = input->height;
    output->depth = input->depth;
    cudaMallocManaged(&output->data, sizeof(float) * output->width * output->height * output->depth);

    for (int i = 0; i < input->width * input->height * input->depth; i++) {
        output->data[i] = input->data[i] > 0 ? input->data[i] : 0;
    }
}

__global__ void add_padding_kernel(img *input, img *output, int padding) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < output->width && y < output->height && z < output->depth) {
        int input_x = x - padding;
        int input_y = y - padding;
        int input_z = z;
        if (input_x >= 0 && input_x < input->width && input_y >= 0 && input_y < input->height) {
            output->data[z * output->width * output->height + y * output->width + x] =
                    input->data[input_z * input->width * input->height + input_y * input->width + input_x];
        } else {
            output->data[z * output->width * output->height + y * output->width + x] = 0.0f;
        }
    }
}

__host__ void add_padding(img *input, img *output, int padding) {
    output->width = input->width + 2 * padding;
    output->height = input->height + 2 * padding;
    output->depth = input->depth;
    cudaMallocManaged(&output->data, sizeof(float) * output->width * output->height * output->depth);

    dim3 block_size(16, 16, 4);
    dim3 num_blocks(ceil(output->width / (float) block_size.x), ceil(output->height / (float) block_size.y),
                    ceil(output->depth / (float) block_size.z));

    add_padding_kernel<<<num_blocks, block_size>>>(input, output, padding);
    cudaDeviceSynchronize();
}

__global__ void dense_kernel(img *input, img *output, float* weights, float* biases) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < output->width) {
        float sum = 0.0f;
        for (int i = 0; i < input->width * input->height * input->depth; i++) {
            sum += input->data[i] * weights[x * input->width * input->height * input->depth + i];
        }
        output->data[x] = sum + biases[x];
    }
}

__host__ void dense(img *input, img *output, float* weights, float* biases, int n_neurons) {
    output->width = n_neurons;
    output->height = 1;
    output->depth = 1;
    cudaMallocManaged(&output->data, sizeof(float) * output->width * output->height * output->depth);

    dense_kernel<<<ceil(output->width / 1024.0f), 1024>>>(input, output, weights, biases);
    cudaDeviceSynchronize();
}

__host__ void dense_cpu(img *input, img *output, float* weights, float* biases, int n_neurons) {
    output->width = n_neurons;
    output->height = 1;
    output->depth = 1;
    cudaMallocManaged(&output->data, sizeof(float) * output->width * output->height * output->depth);

    for (int i = 0; i < output->width; i++) {
        float sum = 0.0f;
        for (int j = 0; j < input->width * input->height * input->depth; j++) {
            sum += input->data[j] * weights[i * input->width * input->height * input->depth + j];
        }
        output->data[i] = sum + biases[i];
    }
}

__host__ void add_padding_cpu(img *input, img *output, int padding) {
    output->width = input->width + 2 * padding;
    output->height = input->height + 2 * padding;
    output->depth = input->depth;
    cudaMallocManaged(&output->data, sizeof(float) * output->width * output->height * output->depth);

    for (int x = 0; x < output->width; x++) {
        for (int y = 0; y < output->height; y++) {
            for (int z = 0; z < output->depth; z++) {
                int input_x = x - padding;
                int input_y = y - padding;
                int input_z = z;
                if (input_x >= 0 && input_x < input->width && input_y >= 0 && input_y < input->height) {
                    output->data[z * output->width * output->height + y * output->width + x] =
                            input->data[input_z * input->width * input->height + input_y * input->width + input_x];
                } else {
                    output->data[z * output->width * output->height + y * output->width + x] = 0.0f;
                }
            }
        }
    }
}

__host__ void print_img(img *input) {
    for (int i = 0; i < input->height; i++) {
        for (int j = 0; j < input->width; j++) {
            for (int k = 0; k < input->depth; k++) {
                printf("%f ", input->data[k * input->width * input->height + i * input->width + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

__host__ void print_ker(ker *kernel) {
    for (int i = 0; i < kernel->size; i++) {
        for (int j = 0; j < kernel->size; j++) {
            for (int k = 0; k < kernel->depth; k++) {
                printf("%f ", kernel->data[k * kernel->size * kernel->size + j * kernel->size + i]);
            }
            printf("\n");
        }
        printf("\n");
    }
}