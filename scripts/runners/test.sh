#!/bin/bash
#SBATCH --job-name=blowup-test
#SBATCH --output=blowup-test-%j.out
#SBATCH --error=blowup-test-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
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

DATAPATH="/cluster/scratch/vogtva/data"

Nx=128
dx=1.0
Nt=10_000
dt=0.0001
n_snapshots=500
model="bruss"
run_id="blowup_test"

mkdir -p $DATAPATH/$model/$run_id

A=11
B=55
Du=1.0
Dv=18.0

for random_seed in 1 2 3 4 5; do
        start=`date +%s`
        FILENAME="${DATAPATH}/${model}/${run_id}/$(uuidgen).nc"
        echo "(A, B) = ($A, $B)"
        FILE=$(python scripts/rd_runner.py --model $model --A $A --B $B \
                --Nx $Nx --dx $dx --Nt $Nt --dt $dt --Du $Du --Dv $Dv \
                --n_snapshots $n_snapshots --filename $FILENAME --run_id=$run_id \
                --random_seed $random_seed)
        build/run_from_netcdf $FILE 1
        end=`date +%s`
        runtime=$((end-start))
        echo "Took $runtime seconds"
done

# Sanitization
# for file in ${DATAPATH}/*; do
#         mv "$file" "$(echo "$file" | sed 's/\xA0//g')"
# done
