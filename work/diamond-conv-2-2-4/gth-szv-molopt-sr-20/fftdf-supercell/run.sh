#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240GB
#SBATCH --reservation=changroup_standingres

# Load environment configuration
source ~/.bashrc
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

conda activate fftisdf

echo ""; which python
python -c "import pyscf; print(pyscf.__version__)"
python -c "import scipy; print(scipy.__version__)"
python -c "import numpy; print(numpy.__version__)"

export PYTHONPATH=/home/junjiey/work/fftisdf-benchmark/src/:$PYTHONPATH;

export PYSCF_EXT_PATH=$HOME/packages/pyscf-forge/pyscf-forge-yangjunjie-non-orth/
cp /home/junjiey/work/fftisdf-benchmark//src/main-fftdf-supercell.py main.py
python main.py --cell=diamond-conv.vasp --kmesh=2-2-4 --basis=gth-szv-molopt-sr --ke_cutoff=20.00 --pseudo=gth-pade 
