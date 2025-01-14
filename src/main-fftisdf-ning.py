import numpy, scipy, os, sys
from pyscf.pbc.scf import KRHF

from pyscf import lib
from pyscf.lib import logger
from pyscf.lib.logger import perf_counter
from pyscf.lib.logger import process_clock

TMPDIR = lib.param.TMPDIR
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
            'prim_cell'
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

def ISDF(cell, kmesh=None, c0=None, _isdf_to_save=None):
    direct = "outcore"
    rela_qr = 1e-4
    aoR_cutoff = 1e-12
    build_V_K_bunchsize = 160

    t0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(cell, 10)

    log.info("ISDF module: %s" % isdf_local_k.__file__)

    if kmesh is None:
        kmesh = [4, 4, 2]

    natm = cell.natm
    partition = [[x] for x in range(natm)]

    isdf_obj = ISDF_Local_K(
        cell.copy(deep=True), kmesh=kmesh, 
        limited_memory=True, direct=direct,
        with_robust_fitting=False,
        build_V_K_bunchsize=build_V_K_bunchsize,
        aoR_cutoff=aoR_cutoff
    )

    isdf_obj.verbose = 10
    isdf_obj._isdf = None
    isdf_obj._isdf_to_save = _isdf_to_save
    isdf_obj.build(c=c0, m=5, rela_cutoff=rela_qr, group=partition)

    log.info("effective c = %6.2f", (float(isdf_obj.naux) / isdf_obj.nao))
    log.timer("ISDF build", *t0)

    import pickle
    with open(isdf_obj._isdf_to_save, "wb") as f:
        pickle.dump(isdf_obj, f)
        log.debug("finished saving to %s", isdf_obj._isdf_to_save)

    isdf_obj._isdf = _isdf_to_save
    return isdf_obj

def main(args):
    from build import cell_from_poscar
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

    stdout = open("out.log", "w")
    log = logger.Logger(stdout, 5)

    kmesh = [int(x) for x in args.kmesh.split("-")]
    kmesh = numpy.array(kmesh)
    nkpt = nimg = numpy.prod(kmesh)
    kpts = cell.get_kpts(kmesh)

    scf_obj = KRHF(cell, kpts=kpts)
    scf_obj.exxdiv = None
    dm_kpts = scf_obj.get_init_guess(key="1e")
    nao = dm_kpts.shape[-1]
    assert dm_kpts.shape == (nkpt, nao, nao)

    t0 = (process_clock(), perf_counter())
    scf_obj.with_df = ISDF(
        cell, kmesh=kmesh, c0=args.c0,
        _isdf_to_save=os.path.join(TMPDIR, "isdf.chk")
    )
    t1 = log.timer("FFTISDF", *t0)

    t0 = (process_clock(), perf_counter())
    vj1, vk1 = scf_obj.get_jk(dm_kpts=dm_kpts, with_j=True, with_k=True)
    vj1 = vj1.reshape(nkpt, nao, nao)
    vk1 = vk1.reshape(nkpt, nao, nao)
    t1 = log.timer("FFTISDF JK", *t0)
    log.info("chk file size: %6.2e GB" % (os.path.getsize(scf_obj.with_df._isdf) / 1e9))

    from pyscf.lib.chkfile import dump
    dump("vjk.chk", "vj", vj1)
    dump("vjk.chk", "vk", vk1)

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
