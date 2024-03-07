#!/bin/bash

# SBATCH --gpus=1

module load gcc/11.4.0
module load cmake/3.26.3
module load cuda/12.1.1
module load hdf5/1.10.9
module load netcdf/4.9.2


cmake -B build -DENABLE_CUDA=1 
make -C build

./build/main
