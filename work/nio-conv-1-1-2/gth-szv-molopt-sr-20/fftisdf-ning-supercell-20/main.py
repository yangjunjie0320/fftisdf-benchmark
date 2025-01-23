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

from pyscf.isdf import isdf_local
class ISDF_Local(isdf_local.ISDF_Local):
    __getstate__ = getstate
    __setstate__ = setstate

class ISDF(object):
    cell = None
    group = None

    kpts = None
    kmesh = None
    c0 = None
    verbose = 10

    _isdf = None
    _isdf_to_save = None
    
    def __init__(self, cell=None, kpts=None):
        assert kpts is None
        self.cell = cell
        self.kpts = kpts

    def build(self):
        cell = self.cell.copy(deep=True)

        group = self.group
        assert group is not None

        direct = False
        c0 = self.c0
        rela_qr = 1e-4
        aoR_cutoff = 1e-12
        build_V_K_bunchsize = 256
        with_robust_fitting = False

        from pyscf.lib import logger
        from pyscf.lib.logger import perf_counter
        from pyscf.lib.logger import process_clock
        t0 = (process_clock(), perf_counter())
        log = logger.new_logger(cell, 10)
        log.info("ISDF module: %s" % isdf_local.__file__)

        isdf_obj = ISDF_Local(
            cell, limited_memory=True, direct=direct,
            with_robust_fitting=with_robust_fitting,
            build_V_K_bunchsize=build_V_K_bunchsize,
        )

        isdf_obj.verbose = 10
        log.info("c0 = %6.2f" % c0)

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

    c = cell_from_poscar(path)
    c.basis = args.basis
    c.pseudo = args.pseudo
    c.verbose = 0
    c.unit = 'aa'
    # c.exp_to_discard = 0.1
    # c.exp_to_discard = 0.0
    c.max_memory = PYSCF_MAX_MEMORY
    c.ke_cutoff = args.ke_cutoff
    c.build(dump_input=False)

    kmesh = [int(x) for x in args.kmesh.split("-")]
    kmesh = numpy.array(kmesh)
    kpts = c.get_kpts(kmesh)

    pcell = c.copy(deep=True)

    natm = len(pcell.atom)
    atm = []
    for ia in range(natm):
        s = pcell.atom_symbol(ia)
        x = pcell.atom_coord(ia, unit='A')
        atm.append([s, x])

    from pyscf.lib.parameters import BOHR
    lvx = pcell.lattice_vectors() * BOHR

    from pyscf.isdf.isdf_tools_cell import build_supercell_with_partition
    scell, group = build_supercell_with_partition(
        atm, lvx, spin=0, charge=0, mesh=None, Ls=kmesh,
        partition=None, basis=args.basis,
        pseudo=args.pseudo, ke_cutoff=args.ke_cutoff,
        max_memory=PYSCF_MAX_MEMORY, 
        precision=1e-12, 
        use_particle_mesh_ewald=True,
        verbose=4,
    )

    from utils import get_jk_time
    df_obj = ISDF(scell)
    df_obj.group = group
    df_obj.c0 = args.c0
    df_obj.verbose = 10
    df_obj.kmesh = kmesh
    df_obj._isdf = os.path.join(TMPDIR, "isdf.chk")

    from utils import get_jk_time
    kmesh = None
    get_jk_time(scell, kmesh=kmesh, df_obj=df_obj, tmp=df_obj._isdf)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=str, default="diamond-prim.vasp")
    parser.add_argument("--kmesh", type=str, default="2-2-2")
    parser.add_argument("--c0", type=float, default=20.0)
    parser.add_argument("--ke_cutoff", type=float, default=40)
    parser.add_argument("--basis", type=str, default="gth-dzvp-molopt-sr")
    parser.add_argument("--pseudo", type=str, default="gth-pade")

    args = parser.parse_args()
    main(args)
