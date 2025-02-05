import mpi4py
from mpi4py import MPI

import os, sys
import numpy, scipy
import scipy.linalg

import pyscf
from pyscf import lib
from pyscf.lib import logger, current_memory
from pyscf.lib.logger import process_clock, perf_counter

from pyscf.pbc.df.fft import FFTDF
from pyscf.pbc import tools as pbctools
from pyscf.pbc.lib.kpts_helper import is_zero

from pyscf.pbc.tools.k2gamma import get_phase
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks

import line_profiler

PYSCF_MAX_MEMORY = int(os.environ.get("PYSCF_MAX_MEMORY", 2000))

import fft_isdf_new

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def build(df_obj, c0=None, kpts=None, kmesh=None):
    """
    Build the FFT-ISDF object.
    
    Args:
        df_obj: The FFT-ISDF object to build.
    """
    log = logger.new_logger(df_obj, df_obj.verbose)
    t0 = (process_clock(), perf_counter())

    cell = df_obj.cell
    assert numpy.allclose(cell.get_kpts(kmesh), kpts)
    nkpt = len(kpts)

    tol = df_obj.tol

    # build the interpolation vectors
    g0 = df_obj.get_inpv(c0=c0)
    nip = g0.shape[0]
    assert g0.shape == (nip, 3)
    nao = cell.nao_nr()

    inpv_kpt = cell.pbc_eval_gto("GTOval", g0, kpts=kpts)
    inpv_kpt = numpy.asarray(inpv_kpt, dtype=numpy.complex128)
    assert inpv_kpt.shape == (nkpt, nip, nao)
    log.debug("nip = %d, cisdf = %6.2f", nip, nip / nao)
    t1 = log.timer("get interpolating vectors")

    coul_kpt = []
    for q in range(nkpt):
        if rank != q % size:
            continue

        t0 = (process_clock(), perf_counter())
        from pyscf.lib import H5TmpFile
        fswp = H5TmpFile()

        from fft_isdf_new import get_lhs_and_rhs
        metx_q, eta_q = get_lhs_and_rhs(
            df_obj, inpv_kpt, kpt=kpts[q], 
            fswp=fswp
        )

        # xi_q: solution for least-squares fitting
        # rho = xi_q * inpv_kpt.conj().T * inpv_kpt
        # but we would not explicitly compute

        ngrid = eta_q.shape[0]
        assert metx_q.shape == (nip, nip)
        assert eta_q.shape == (ngrid, nip)

        from fft_isdf_new import get_coul
        kern_q = get_coul(
            df_obj, eta_q, kpt=kpts[q], 
            tol=tol, fswp=fswp
        )

        from fft_isdf_new import lstsq
        res = lstsq(metx_q, kern_q, tol=tol)
        coul_q = res[0]
        assert coul_q.shape == (nip, nip)

        coul_kpt.append((q, coul_q))
        log.timer("solving Coulomb kernel", *t0)
        # print("q = %d, rank = %d / %d" % (q, rank, size))
        log.info("Finished solving Coulomb kernel for q = %3d / %3d, rank = %d / %d", q + 1, nkpt, res[1], nip)

    comm.barrier()
    coul_kpt = comm.allreduce(coul_kpt)
    coul_kpt = [x[1] for x in sorted(coul_kpt, key=lambda x: x[0])]
    coul_kpt = numpy.asarray(coul_kpt)
    coul_kpt = coul_kpt.reshape(nkpt, nip, nip)
    comm.barrier()
    return inpv_kpt, coul_kpt

fft_isdf_new.build = build

class WithMPI(fft_isdf_new.FFTISDF):
    pass

FFTISDF = ISDF = WithMPI

if __name__ == "__main__":
    DATA_PATH = os.getenv("DATA_PATH", "../data/")
    from utils import cell_from_poscar

    TMPDIR = lib.param.TMPDIR

    cell = cell_from_poscar(os.path.join(DATA_PATH, "diamond-prim.vasp"))
    cell.basis = 'gth-dzvp-molopt-sr'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.unit = 'aa'
    cell.exp_to_discard = 0.1
    cell.max_memory = PYSCF_MAX_MEMORY
    cell.ke_cutoff = 80.0
    cell.stdout = sys.stdout if rank == 0 else open(TMPDIR + "out-%d.log" % rank, "w")
    cell.build(dump_input=False)

    nao = cell.nao_nr()

    # kmesh = [4, 4, 4]
    kmesh = [4, 4, 4]
    nkpt = nspc = numpy.prod(kmesh)
    kpts = cell.get_kpts(kmesh)

    scf_obj = pyscf.pbc.scf.KRHF(cell, kpts=kpts)
    scf_obj.exxdiv = None
    scf_obj.stdout = cell.stdout
    dm_kpts = scf_obj.get_init_guess(key="minao")

    log = logger.new_logger(cell, 5)
    log.stdout = cell.stdout

    t0 = (process_clock(), perf_counter())
    scf_obj.with_df = FFTDF(cell, kpts)
    scf_obj.with_df.verbose = 5
    scf_obj.with_df.stdout = cell.stdout

    if rank == 0:
        scf_obj.with_df.dump_flags()
        scf_obj.with_df.check_sanity()

    vj0 = numpy.zeros((nkpt, nao, nao))
    vk0 = numpy.zeros((nkpt, nao, nao))
    vj0, vk0 = scf_obj.get_jk(dm_kpts=dm_kpts, with_j=True, with_k=True)
    vj0 = vj0.reshape(nkpt, nao, nao)
    vk0 = vk0.reshape(nkpt, nao, nao)

    if rank == 0:
        t1 = log.timer("-> FFTDF JK", *t0)

    for c0 in [5.0, 10.0, 15.0, 20.0]:
        comm.barrier()

        t0 = (process_clock(), perf_counter())
        scf_obj.with_df = ISDF(cell, kpts=kpts)
        scf_obj.with_df.c0 = c0
        scf_obj.with_df.verbose = 5
        scf_obj.with_df.tol = 1e-12
        df_obj = scf_obj.with_df
        df_obj.build()

        t1 = log.timer("-> ISDF build", *t0)

        t0 = (process_clock(), perf_counter())
        vj1 = numpy.zeros((nkpt, nao, nao))
        vk1 = numpy.zeros((nkpt, nao, nao))
        vj1, vk1 = scf_obj.get_jk(dm_kpts=dm_kpts, with_j=True, with_k=True)
        vj1 = vj1.reshape(nkpt, nao, nao)
        vk1 = vk1.reshape(nkpt, nao, nao)

        if rank == 0:
            t1 = log.timer("-> ISDF JK", *t0)

            err = abs(vj0 - vj1).max()
            print("-> ISDF c0 = % 6.2f, vj err = % 6.4e" % (c0, err))

            err = abs(vk0 - vk1).max()
            print("-> ISDF c0 = % 6.2f, vk err = % 6.4e" % (c0, err))

        comm.barrier()