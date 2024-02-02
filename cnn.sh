#!/usr/bin/env zsh

#SBATCH --job-name=cnn
#SBATCH --partition=instruction
#SBATCH --time=00-00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=cnn.out

cd $SLURM_SUBMIT_DIR

module load nvidia/cuda/11.8.0

# nvcc convolution.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o convolution

nvcc cnn.cu convolution.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o cnn

# ./cnn

for i in {1..7}
do
    n=$(((2*i)+1))
    ./cnn 1024 $n
done
