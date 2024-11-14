#!/bin/bash
#SBATCH --job-name=bruss
#SBATCH --output=bruss-%j.out
#SBATCH --error=bruss-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=2
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=00:15:00

module load stack/2024-06
module load gcc/12.2.0
module load cmake/3.27.7
module load cuda/12.1.1
module load hdf5/1.14.3
module load openmpi/4.1.6
module load netcdf-c/4.9.2
module load python/3.11.6

# A: float = 5
# B: float = 9
# Nx: int = 100
# dx: float = 1.0
# Nt: int = 1000
# dt: float = 0.01
# Du: float = 2.0
# Dv: float = 22.0
# n_snapshots: int = 100

A=5
B=9
Nx=100
dx=1.0
Nt=1000
dt=0.01
Du=2.0
Dv=22.0
n_snapshots=100

FILE=$(python3 scripts/rd_runner.py --model bruss --A $A --B $B \
        -Nx $Nx --dx $dx --Nt $Nt --dt $dt --Du $Du --Dv $Dv \
        --n_snapshots $n_snapshots)

build/run_from_netcdf $FILE
