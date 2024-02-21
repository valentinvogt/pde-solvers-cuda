#!/bin/bash

module load gcc/11.4.0
module load cmake/3.26.3
module load cuda/12.1.1

cmake -B build
make -C build

./build/main
