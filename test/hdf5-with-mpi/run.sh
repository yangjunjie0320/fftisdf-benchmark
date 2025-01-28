#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --reservation=changroup_standingres

# Load environment configuration
source /home/junjiey/anaconda3/bin/activate fftisdf-with-mpi
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

echo ""; which python
python -c "import pyscf; print(pyscf.__version__)"
python -c "import scipy; print(scipy.__version__)"
python -c "import numpy; print(numpy.__version__)"

echo "MPI version: $(mpicc --version)"
echo "HDF5 version: $(h5c --version)"
echo "NPROCS = $SLURM_NPROCS"

export PYTHONPATH=/home/junjiey/work/fftisdf-benchmark/src/:$PYTHONPATH;
export PYSCF_EXT_PATH=$HOME/packages/pyscf-forge/pyscf-forge-yangjunjie-non-orth/
mpirun -n $SLURM_NPROCS python main.py
