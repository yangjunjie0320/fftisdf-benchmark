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
    cell.ke_cutoff = args.ke_cutoff
    cell.build(dump_input=False)

    kmesh = [int(x) for x in args.kmesh.split("-")]
    kmesh = numpy.array(kmesh)
    kpts = cell.get_kpts(kmesh)

    from fft_isdf import FFTISDF
    df_obj = FFTISDF(cell, kpts=kpts)
    df_obj.c0 = args.c0
    df_obj.tol = 1e-8
    df_obj.verbose = 10
    df_obj._isdf = os.path.join(TMPDIR, "isdf-old.chk")
    df_obj._isdf_to_save = os.path.join(TMPDIR, "isdf-old.chk")

    from utils import get_jk_time
    get_jk_time(
        cell, kmesh, df_obj=df_obj, tmp=df_obj._isdf, 
        chkfile=os.path.join(TMPDIR, "vjk-old.chk"),
        stdout=open("out-old.log", "w")
    )

    from fft_isdf_new import FFTISDF
    df_obj = FFTISDF(cell, kpts=kpts)
    df_obj.c0 = args.c0
    df_obj.tol = 1e-8
    df_obj.verbose = 10
    df_obj._isdf = os.path.join(TMPDIR, "isdf-new.chk")
    df_obj._isdf_to_save = os.path.join(TMPDIR, "isdf-new.chk")

    from utils import get_jk_time
    get_jk_time(
        cell, kmesh, df_obj=df_obj, tmp=df_obj._isdf, 
        chkfile=os.path.join(TMPDIR, "vjk-new.chk"),
        stdout=open("out-new.log", "w")
    )

    from pyscf.pbc.df.fft import FFTDF
    df_obj = FFTDF(cell, kpts=kpts)
    df_obj.verbose = 10

    from utils import get_jk_time
    tmp = os.path.join(TMPDIR, "fftdf.chk")
    os.system("touch %s" % tmp)

    get_jk_time(
        cell, kmesh, df_obj=df_obj, tmp=tmp, 
        chkfile=os.path.join(TMPDIR, "vjk-ref.chk"),
        stdout=open("out-ref.log", "w")
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=str, default="diamond-prim.vasp")
    parser.add_argument("--kmesh", type=str, default="2-2-2")
    parser.add_argument("--c0", type=float, default=20.0)
    parser.add_argument("--ke_cutoff", type=float, default=200)
    parser.add_argument("--basis", type=str, default="gth-dzvp-molopt-sr")
    parser.add_argument("--pseudo", type=str, default="gth-pade")

    args = parser.parse_args()
    main(args)
