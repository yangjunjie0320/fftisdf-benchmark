import os, sys
import numpy, scipy
import pyscf

TMPDIR = os.getenv("TMPDIR", None)
DATA_PATH = os.getenv("DATA_PATH", None)
PYSCF_MAX_MEMORY = os.getenv("PYSCF_MAX_MEMORY", 4000)
PYSCF_MAX_MEMORY = int(PYSCF_MAX_MEMORY)

try:
    from pyscf.lib import generate_pickle_methods
    getstate, setstate = generate_pickle_methods(
        excludes=(
            '_isdf_to_save', 
            '_isdf', 
            'buffer_fft',
            'buffer_cpu', 
            'buffer_gpu', 
            'buffer',
            'cell', 
            'prim_cell',
            '_swapfile',
        )
    )   
except ImportError:
    import sys 
    sys.stderr.write("pyscf.lib.generate_pickle_methods is not available, ISDF will not support pickle\n")
    def raise_error(*args, **kwargs):
        raise NotImplementedError("ISDF does not support pickle")
    getstate = setstate = raise_error

from pyscf.isdf import isdf_local_k
class ISDF_Local_K(isdf_local_k.ISDF_Local_K):
    __getstate__ = getstate
    __setstate__ = setstate

class ISDF(object):
    cell = None
    kpts = None
    kmesh = None
    c0 = None
    verbose = 10

    _isdf = None
    _isdf_to_save = None
    
    def __init__(self, cell, kpts=None):
        self.cell = cell
        self.kpts = kpts

    def build(self):
        cell = self.cell.copy(deep=True)
        kmesh = self.kmesh
        c0 = self.c0

        direct = "outcore"
        rela_qr = 1e-4
        aoR_cutoff = 1e-12
        build_V_K_bunchsize = 32
        with_robust_fitting = False

        from pyscf.lib import logger
        from pyscf.lib.logger import perf_counter
        from pyscf.lib.logger import process_clock
        t0 = (process_clock(), perf_counter())
        log = logger.new_logger(cell, 10)
        log.info("ISDF module: %s" % isdf_local_k.__file__)

        isdf_obj = ISDF_Local_K(
            cell, kmesh=kmesh, direct=direct,
            with_robust_fitting=with_robust_fitting,
            limited_memory=True, aoR_cutoff=aoR_cutoff,
            build_V_K_bunchsize=build_V_K_bunchsize,
        )

        isdf_obj.verbose = 10
        log.info("c0 = %6.2f" % c0)

        group = [[i] for i in range(isdf_obj.first_natm)]
        isdf_obj.build(c=c0, rela_cutoff=rela_qr, group=group)

        nip = isdf_obj.naux
        log.info(
            "Number of interpolation points = %d, effective CISDF = %6.2f",
            nip, nip / isdf_obj.nao
        )
        log.timer("ISDF build", *t0)

        import pickle
        with open(self._isdf, "wb") as f:
            pickle.dump(isdf_obj, f)
            log.debug("finished saving to %s", self._isdf)

        isdf_obj._isdf = self._isdf
        isdf_obj._isdf_to_save = self._isdf
        return isdf_obj

def main(args):
    from utils import cell_from_poscar
    path = os.path.join(DATA_PATH, args.cell)
    assert os.path.exists(path), "Cell file not found: %s" % path

    cell = cell_from_poscar(path)
    cell.basis = args.basis
    cell.pseudo = args.pseudo
    cell.verbose = 0
    cell.unit = 'aa'
    cell.exp_to_discard = 0.1
    cell.max_memory = PYSCF_MAX_MEMORY
    cell.ke_cutoff = args.ke_cutoff
    cell.build(dump_input=False)

    kmesh = [int(x) for x in args.kmesh.split("-")]
    kmesh = numpy.array(kmesh)
    kpts = cell.get_kpts(kmesh)

    from utils import get_jk_time
    df_obj = ISDF(cell, kpts=kpts)
    df_obj.c0 = args.c0
    df_obj.verbose = 10
    df_obj.kmesh = kmesh
    df_obj._isdf = os.path.join(TMPDIR, "isdf.chk")

    from utils import get_jk_time
    get_jk_time(cell, kmesh, df_obj=df_obj, tmp=df_obj._isdf)

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
