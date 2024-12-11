#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=eval-%j.out
#SBATCH --error=eval-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --time=00:30:00

module load stack/2024-06
module load gcc/12.2.0
module load cmake/3.27.7
module load cuda/12.1.1
module load hdf5/1.14.3
module load openmpi/4.1.6
module load netcdf-c/4.9.2
module load python/3.11.6

run_id="ab_new"
var1="A"
var2="B"
file="out/bruss_ab_new.gif"
python3 analysis/anim_db.py --model bruss --subdir $run_id --run_id $run_id --var1 $var1 --var2 $var2 --outfile $file --dpi 150
