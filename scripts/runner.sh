#!/bin/bash
#SBATCH --job-name=time-run
#SBATCH --output=time-run-%j.out
#SBATCH --error=time-run-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=01:00:00

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
Nx=400
dx=0.25
Nt=2_500
dt=0.0025
Du=2.0
Dv=22.0
n_snapshots=100

for dt in 0.001 0.0025 0.005 0.01; do
        FILE=$(python3 scripts/rd_runner.py --model bruss --A $A --B $B \
                --Nx $Nx --dx $dx --Nt $Nt --dt $dt --Du $Du --Dv $Dv \
                --n_snapshots $n_snapshots)

        build/run_from_netcdf $FILE 1
done
