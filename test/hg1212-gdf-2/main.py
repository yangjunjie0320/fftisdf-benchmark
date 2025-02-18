import os, sys
import time

from libdmet.utils.iotools import read_poscar
import libdmet.utils.logger as log

TMPDIR = os.getenv("TMPDIR", None)
DATA_PATH = os.getenv("DATA_PATH", None)
PYSCF_MAX_MEMORY = os.getenv("PYSCF_MAX_MEMORY", 4000)
PYSCF_MAX_MEMORY = int(PYSCF_MAX_MEMORY)

log.verbose = "DEBUG1"

start = time.time()

# this is the number of doped holes
nelec_dop = 0

cell = read_poscar(fname="./Hg1212-2x2.vasp")
cell.basis = 'cc-pvdz.dat'
cell.pseudo = {
            'Cu1': 'GTH-PBE-q19', 
            'Cu2': 'GTH-PBE-q19',
            'O1': 'gth-pbe',
            'O2': 'gth-pbe',
            'Cu3': 'GTH-PBE-q19', 
            'Cu4': 'GTH-PBE-q19',
            'O3': 'gth-pbe',
            'O4': 'gth-pbe',
            'Hg': 'gth-pbe', 
            'Ba': 'gth-pbe',
            'Ca': 'gth-pbe'}

kmesh = [1, 1, 1]
cell.spin = 0
cell.verbose = 5
cell.precision = 1e-8
# cell.ke_cutoff = ke_cutoff
cell.max_memory = PYSCF_MAX_MEMORY
cell.build()

kpts = cell.get_kpts(kmesh)

from pyscf.pbc.df import GDF
df_obj = GDF(cell, kpts=kpts)
df_obj.verbose = 10
df_obj._cderi = "/central/scratch/yangjunjie/hg1212-gdf.h5"
# df_obj._cderi_to_save = df_obj._cderi
# df_obj.build = lambda: None

from utils import scf
scf(cell, kmesh=kmesh, df_obj=df_obj, tmp=TMPDIR, chkfile="hg1212-gdf.chk", read_dm_from="/central/home/junjiey/work/fftisdf-benchmark-new/test/hg1212.chk")
