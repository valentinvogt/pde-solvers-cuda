#!/bin/bash
#SBATCH --job-name=vary-sigma
#SBATCH --output=vary-sigma-%j.out
#SBATCH --error=vary-sigma-%j.err
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
Nx=200
dx=1.0
Nt=1_000
dt=0.0025
Du=2.0
Dv=22.0
n_snapshots=100

#  0.2 0.3 0.4 0.5
for sigma_ic in 0.1; do
        FILENAME="data/vary_sigma_ic/bruss_sigma_${sigma_ic}.nc"
        # echo "$(python3 -c "print(repr('$FILENAME'))")"
        FILENAME=$(echo $FILENAME | tr -d '[:space:]')
        # echo "$(python3 -c "print(repr('$FILENAME'))")"
        
        FILE=$(python3 scripts/rd_runner.py --model bruss --A $A --B $B \
                --Nx $Nx --dx $dx --Nt $Nt --dt $dt --Du $Du --Dv $Dv --sigma_ic $sigma_ic \
                --n_snapshots $n_snapshots --filename $FILENAME)

        build/run_from_netcdf $FILE 1
done