# Implementing Convolutional Neural Networks in Parallel using CUDA

This is a implementation of a parallelized forward pass for a custom Convolutional Neural Network (CNN) using CUDA in C++, aimed at optimizing runtime performance in deep learning models. With a focus on enhancing inference speed, this project endeavored to distribute computation tasks using GPUs to expedite the processing of large datasets and complex CNN architectures.

I was able to establish my approach’s superiority over PyTorch, in inference speed, in all configurations of the CNN. I also got as much as 10X speedup, in some cases, compared to C++ CPU (non-parallel) implementation.
