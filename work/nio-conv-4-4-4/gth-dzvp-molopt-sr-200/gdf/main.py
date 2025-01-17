import os, sys
import numpy, scipy
import pyscf

TMPDIR = os.getenv("TMPDIR", None)
DATA_PATH = os.getenv("DATA_PATH", None)
PYSCF_MAX_MEMORY = os.getenv("PYSCF_MAX_MEMORY", 4000)
PYSCF_MAX_MEMORY = int(PYSCF_MAX_MEMORY)

def main(args):
    from utils import cell_from_poscar
    path = os.path.join(DATA_PATH, args.cell)
    assert os.path.exists(path), "Cell file not found: %s" % path

    cell = cell_from_poscar(path)
    cell.basis = args.basis
    cell.pseudo = args.pseudo
    cell.verbose = 5
    cell.unit = 'aa'
    cell.exp_to_discard = 0.1
    cell.max_memory = PYSCF_MAX_MEMORY
    cell.ke_cutoff = args.ke_cutoff
    cell.build(dump_input=False)

    kmesh = [int(x) for x in args.kmesh.split("-")]
    kmesh = numpy.array(kmesh)
    kpts = cell.get_kpts(kmesh)

    from pyscf.pbc.df import GDF
    df_obj = GDF(cell, kpts=kpts)
    df_obj.verbose = 10
    df_obj._cderi_to_save = None
    df_obj._cderi = "/central/scratch/yangjunjie//gdf/46829530/gdf.chk"

    from utils import get_jk_time
    get_jk_time(cell, kmesh, df_obj=df_obj, tmp=df_obj._cderi)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=str, default="diamond-prim.vasp")
    parser.add_argument("--kmesh", type=str, default="2-2-2")
    parser.add_argument("--ke_cutoff", type=float, default=200)
    parser.add_argument("--basis", type=str, default="gth-dzvp-molopt-sr")
    parser.add_argument("--pseudo", type=str, default="gth-pade")

    args = parser.parse_args()
    main(args)
