#!/bin/bash
#SBATCH --job-name=ball
#SBATCH --output=ball-%j.out
#SBATCH --error=ball-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=4096
#SBATCH --time=1:00:00

module load stack/2024-06
module load gcc/12.2.0
module load cmake/3.27.7
module load cuda/12.1.1
module load hdf5/1.14.3
module load openmpi/4.1.6
module load netcdf-c/4.9.2
module load python/3.11.6

./build/run_from_netcdf data/test.nc 1
