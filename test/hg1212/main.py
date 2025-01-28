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
cell.ke_cutoff = 200
cell.build()

kpts = cell.get_kpts(kmesh)

from fft_isdf import ISDF
df_obj = ISDF(cell, kpts=kpts)
df_obj.c0 = 20.0
df_obj.tol = 1e-16
df_obj.verbose = 10
df_obj._isdf = os.path.join(TMPDIR, "tmp.chk")
df_obj.build()
