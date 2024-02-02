#!/usr/bin/env zsh

#SBATCH --job-name=convolution
#SBATCH --partition=instruction
#SBATCH --time=00-00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=convolution.out

cd $SLURM_SUBMIT_DIR

module load nvidia/cuda/11.8.0

nvcc convolution.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o convolution

./convolution
