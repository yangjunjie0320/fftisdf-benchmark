import numpy, scipy, os, sys
from pyscf.pbc.scf import KRHF

from pyscf.lib import logger
from pyscf.pbc.df import FFTDF, GDF
from pyscf.lib.logger import perf_counter
from pyscf.lib.logger import process_clock

from fft_isdf import ISDF

DATA_PATH = os.getenv("DATA_PATH", None)
PYSCF_MAX_MEMORY = os.getenv("PYSCF_MAX_MEMORY", 4000)
PYSCF_MAX_MEMORY = int(PYSCF_MAX_MEMORY)

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
    cell.output = "out.log"
    cell.build(dump_input=False)

    log = logger.new_logger(sys.stdout, 5)
    kmesh = [int(x) for x in args.kmesh.split("-")]
    kmesh = numpy.array(kmesh)
    nkpt = nimg = numpy.prod(kmesh)
    kpts = cell.get_kpts(kmesh)

    scf_obj = KRHF(cell, kpts=kpts)
    scf_obj.exxdiv = None
    dm_kpts = scf_obj.get_init_guess(key="1e")
    nao = dm_kpts.shape[-1]
    assert dm_kpts.shape == (nkpt, nao, nao)

    scf_obj.with_df = ISDF(cell, kpts=kpts)

    t0 = (process_clock(), perf_counter())
    c0 = args.c0
    m0 = [int(x) for x in args.m0.split("-")]
    m0 = numpy.array(m0)
    scf_obj.with_df.c0 = c0
    scf_obj.with_df.m0 = m0
    scf_obj.with_df.verbose = 10
    scf_obj.with_df.tol = 1e-10
    scf_obj.with_df.build()
    t1 = log.timer("build ISDF", *t0)

    t0 = (process_clock(), perf_counter())
    vj1, vk1 = scf_obj.get_jk(dm_kpts=dm_kpts, with_j=True, with_k=True)
    vj1 = vj1.reshape(nkpt, nao, nao)
    vk1 = vk1.reshape(nkpt, nao, nao)
    t1 = log.timer("ISDF JK", *t0)

    t0 = (process_clock(), perf_counter())
    if args.reference.lower() == "fft":
        scf_obj.with_df = FFTDF(cell, kpts)
    elif args.reference.lower() == "gdf":
        scf_obj.with_df = GDF(cell, kpts)
    else:
        raise ValueError("Invalid reference: %s" % args.reference)
    
    t0 = (process_clock(), perf_counter())
    scf_obj.with_df.verbose = 200
    scf_obj.with_df.dump_flags()
    scf_obj.with_df.check_sanity()
    scf_obj.with_df.build()
    t1 = log.timer("build %s" % args.reference, *t0)

    vj0, vk0 = scf_obj.get_jk(dm_kpts=dm_kpts, with_j=True, with_k=True)
    vj0 = vj0.reshape(nkpt, nao, nao)
    vk0 = vk0.reshape(nkpt, nao, nao)
    t1 = log.timer("%s JK" % args.reference, *t0)

    err = abs(vj0 - vj1).max()
    print("c0 = % 6.2f, vj err = % 6.4e" % (args.reference, c0, err))

    err = abs(vk0 - vk1).max()
    print("c0 = % 6.2f, vk err = % 6.4e" % (args.reference, c0, err))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=str, default="diamond-prim.vasp")
    parser.add_argument("--kmesh", type=str, default="2-2-2")
    parser.add_argument("--c0", type=float, default=20.0)
    parser.add_argument("--m0", type=str, default="19-19-19")
    parser.add_argument("--reference", type=str, default="fft")
    parser.add_argument("--ke_cutoff", type=float, default=200)
    parser.add_argument("--basis", type=str, default="gth-dzvp-molopt-sr")
    parser.add_argument("--pseudo", type=str, default="gth-pade")

    args = parser.parse_args()
    main(args)
