#!/bin/bash
#SBATCH --job-name=ball
#SBATCH --output=ball-%j.out
#SBATCH --error=ball-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=4096
#SBATCH --time=8:00:00
#SBATCH --mail-type=END

module load stack/2024-06
module load gcc/12.2.0
module load cmake/3.27.7
module load cuda/12.1.1
module load hdf5/1.14.3
module load openmpi/4.1.6
module load netcdf-c/4.9.2
module load python/3.11.6

DATAPATH="/cluster/scratch/vogtva/data"

# ADAPT THESE
model="gray_scott"
run_id="new_ball"

# step=15
# count=0

# rm -f "$DATAPATH/$model/$run_id/"*_output.nc

for file in "$DATAPATH/$model/$run_id"/*.nc; do
    build/run_from_netcdf $file 1
    # ((count++))
    # if ((count % step == 0)); then
    #     wait
    # fi
done

# for json_file in "$DATAPATH/$model/$run_id"/*.json; do
#     base_name="${json_file%.json}"  # Remove .json extension
#     nc_file="${base_name}_output.nc"

#     if [[ ! -f "$nc_file" ]]; then
#         echo "Running for: $base_name".nc
#         build/run_from_netcdf "$base_name".nc 1
#     fi
# done
