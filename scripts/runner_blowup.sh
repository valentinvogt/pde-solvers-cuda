#!/bin/bash
#SBATCH --job-name=vary-both
#SBATCH --output=vary-both-%j.out
#SBATCH --error=vary-both-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=04:00:00

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

DATAPATH="/cluster/scratch/vogtva/data/vb-high"

# A=5
# B=9
Nx=400
dx=0.5
Nt=1_000
dt=0.0025
Du=2.0
Dv=22.0
n_snapshots=250

for A in 10 11 12 13 14 15; do
        for mult in 1.6 1.8 2.0 2.2; do
                B=$(python3 -c "print($A * $mult)")
                FILENAME="${DATAPATH}/bruss_A_${A}_B_${B}.nc"
                FILE=$(python3 scripts/rd_runner.py --model bruss --A $A --B $B \
                        --Nx $Nx --dx $dx --Nt $Nt --dt $dt --Du $Du --Dv $Dv \
                        --n_snapshots $n_snapshots --filename $FILENAME)
                build/run_from_netcdf $FILE 1
        done
done

# Sanitization
for file in ${DATAPATH}/*; do
        mv "$file" "$(echo "$file" | sed 's/\xA0//g')"
done