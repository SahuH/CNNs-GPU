#!/usr/bin/env zsh

#SBATCH --job-name=benchCPU
#SBATCH --partition=instruction
#SBATCH --time=00-00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=benchCPU.out

cd $SLURM_SUBMIT_DIR

module load nvidia/cuda/11.8.0

nvcc benchmarkModelsCPU.cu convolution.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o benchmarkModelsCPU

# "Usage: ./benchmarkModelsCPU <input_size> <kernel_size> <n_filters> <n_conv_layers> <n_pooling_layers> <n_dense_layers> <n_neurons>"
./benchmarkModelsCPU 32 3 32 1 1 2 128
#for i in {1..5}
#do
#        n=$(((2*i)+1))
#        ./benchmarkModelsCPU 32 $n 32 1 1 2 128
#done

