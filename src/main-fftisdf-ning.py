import os, sys, pyscf, numpy, scipy
from pyscf import lib 
from pyscf.lib import logger
from pyscf.isdf import isdf_local_k

TMPDIR = pyscf.lib.param.TMPDIR

def ISDF(cell, kmesh=None, cisdf=None, rela_qr=1e-4, 
         with_robust_fitting=True, direct=True,
         build_V_K_bunchsize=28, chkfile=None):
    direct = "outcore"
    with_robust_fitting = False

    t0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(cell, 10)

    # print the isdf module information
    from pyscf import isdf
    import pyscf.isdf.isdf_local_k
    log.info("ISDF module: %s" % pyscf.isdf.isdf_local_k.__file__)

    if kmesh is None:
        kmesh = [4, 4, 2]

    natm = cell.natm
    partition = [[x] for x in range(natm)]

    isdf_obj = ISDF_Local_K(
        cell.copy(deep=True), kmesh=kmesh, 
        limited_memory=True, direct=direct,
        with_robust_fitting=with_robust_fitting,
        build_V_K_bunchsize=build_V_K_bunchsize,
        aoR_cutoff=1e-12
    )

    isdf_obj.verbose = 10
    pc = isdf_obj.prim_cell
    sc = isdf_obj.cell

    if chkfile is not None:
        assert os.path.isfile(chkfile)
        isdf_obj = None
        log.debug("reading from %s", chkfile)

        import pickle
        with open(chkfile, "rb") as f:
            isdf_obj = pickle.load(f)
            log.debug("finished reading from %s", chkfile)
                                                                                                                                                                     
        isdf_obj.prim_cell = pc
        isdf_obj.cell = sc

        assert isdf_obj is not None
        isdf_obj._build_buffer(c=cisdf, m=5, group=partition)
        isdf_obj._build_fft_buffer()

    else:
        isdf_obj._isdf = None
        isdf_obj._isdf_to_save = os.path.join(TMPDIR, "isdf.chk")
        isdf_obj.build(c=cisdf, m=5, rela_cutoff=rela_qr, group=partition)

        import pickle
        with open(isdf_obj._isdf_to_save, "wb") as f:
            pickle.dump(isdf_obj, f)
            log.debug("finished saving to %s", isdf_obj._isdf_to_save)

    log.info("effective c = %6.2f", (float(isdf_obj.naux) / isdf_obj.nao))
    log.timer("ISDF build", *t0)

    isdf_obj.chkfile = chkfile
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
    log = logger.new_logger(stdout, 5)
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
    print("%s: c0 = % 6.2f, vj err = % 6.4e" % (args.reference, c0, err))

    err = abs(vk0 - vk1).max()
    print("%s: c0 = % 6.2f, vk err = % 6.4e" % (args.reference, c0, err))


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
