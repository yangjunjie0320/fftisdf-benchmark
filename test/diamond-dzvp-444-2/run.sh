#!/bin/bash
#SBATCH --job-name=diamond-dzvp-444
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB 
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --reservation=changroup_standingres
#SBATCH --ntasks-per-node=1

# Load environment configuration
source /home/junjiey/anaconda3/bin/activate fftisdf
export DATA_PATH=/home/junjiey/work/fftisdf-benchmark/data/

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
export MKL_NUM_THREADS=1;

export PYSCF_MAX_MEMORY=$SLURM_MEM_PER_NODE;
echo OMP_NUM_THREADS = $OMP_NUM_THREADS
echo MKL_NUM_THREADS = $MKL_NUM_THREADS
echo PYSCF_MAX_MEMORY = $PYSCF_MAX_MEMORY

export TMP=/central/scratch/yangjunjie/
export TMPDIR=$TMP/$SLURM_JOB_NAME/$SLURM_JOB_ID/
export PYSCF_TMPDIR=$TMPDIR

mkdir -p $TMPDIR
echo TMPDIR       = $TMPDIR
echo PYSCF_TMPDIR = $PYSCF_TMPDIR
unlink tmp; ln -s $PYSCF_TMPDIR tmp

echo ""; which python
python -c "import pyscf; print(pyscf.__version__)"
python -c "import scipy; print(scipy.__version__)"
python -c "import numpy; print(numpy.__version__)"

export PYTHONPATH=/home/junjiey/work/fftisdf-benchmark-new/src/:$PYTHONPATH;
# python main.py --c0=10.00 --ke_cutoff=40.00 --cell=diamond-prim.vasp --kmesh=4-4-4 --basis=gth-szv-molopt-sr --pseudo=gth-pade

cp /home/junjiey/work/fftisdf-benchmark-new/src/fft_isdf_new.py main.py
kernprof -l main.py
