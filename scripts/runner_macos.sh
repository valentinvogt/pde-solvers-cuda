#SBATCH --job-name=vary-dx
#SBATCH --output=vary-dx-%j.out
#SBATCH --error=vary-dx-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=01:00:00

# module load stack/2024-06
# module load gcc/12.2.0
# module load cmake/3.27.7
# module load cuda/12.1.1
# module load hdf5/1.14.3
# module load openmpi/4.1.6
# module load netcdf-c/4.9.2
# module load python/3.11.6

# A: float = 5
# B: float = 9
# Nx: int = 100
# dx: float = 1.0
# Nt: int = 1000
# dt: float = 0.01
# Du: float = 2.0
# Dv: float = 22.0
# n_snapshots: int = 100

DATAPATH="data/vary-both"
PYTHON=".venv/bin/python3"

A=0.037
B=0.06
Nx=100
dx=1.0
Nt=10_001
dt=0.05
Du=0.2
Dv=0.1
n_snapshots=100

for sparsity in 1 2 5 10 50; do
        FILENAME="data/gs_sp/gs-sparsity_${sparsity}.nc"
        FILE=$(.venv/bin/python3 scripts/rd_runner.py --model gray_scott --A $A --B $B \
                --Nx $Nx --dx $dx --Nt $Nt --dt $dt --Du $Du --Dv $Dv \
                --n_snapshots $n_snapshots --filename $FILENAME --sparsity $sparsity)
        build/run_from_netcdf $FILENAME
done

# Sanitization
# for file in ${DATAPATH}/*; do
#         mv "$file" "$(echo "$file" | sed 's/\xA0//g')"
# done
