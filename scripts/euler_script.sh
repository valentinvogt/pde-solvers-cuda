#!/bin/bash

#SBATCH --gpus=1

module load gcc/11.4.0
module load cmake/3.26.3
module load cuda/12.1.1


cmake -B build -DENABLE_CUDA=1 
make -C build

./build/main
