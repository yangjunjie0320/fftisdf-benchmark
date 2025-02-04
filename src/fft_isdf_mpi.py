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
print("rank = %d, size = %d" % (rank, size))
comm.Barrier()

def build(df_obj, c0=None, kpts=None, kmesh=None):
    """
    Build the FFT-ISDF object.
    
    Args:
        df_obj: The FFT-ISDF object to build.
    """
    log = logger.new_logger(df_obj, df_obj.verbose)

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

    coul_kpt = []
    for q in range(nkpt):
        if rank != q % size:
            continue

        print("q = %d, rank = %d" % (q, rank))

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
        log.info("Finished solving Coulomb kernel for q = %3d / %3d, rank = %d / %d", q + 1, nkpt, res[1], nip)

    comm.Barrier()
    if rank == 0:
        coul_kpt = comm.gather(coul_kpt, root=0)
        coul_kpt = sorted(coul_kpt, key=lambda x: x[0])
        coul_kpt = [x[1] for x in coul_kpt]
        coul_kpt = numpy.asarray(coul_kpt)

    coul_kpt = numpy.asarray(coul_kpt)
    coul_kpt = coul_kpt.reshape(nkpt, nip, nip)
    return inpv_kpt, coul_kpt

fft_isdf_new.build = build

class WithMPI(fft_isdf_new.FFTISDF):
    pass

FFTISDF = ISDF = WithMPI

if __name__ == "__main__":
    DATA_PATH = os.getenv("DATA_PATH", "../data/")
    from utils import cell_from_poscar

    cell = cell_from_poscar(os.path.join(DATA_PATH, "diamond-prim.vasp"))
    cell.basis = 'gth-dzvp-molopt-sr'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.unit = 'aa'
    cell.exp_to_discard = 0.1
    cell.max_memory = PYSCF_MAX_MEMORY
    cell.ke_cutoff = 80.0
    cell.build(dump_input=False)
    nao = cell.nao_nr()

    kmesh = [4, 4, 4]
    nkpt = nimg = numpy.prod(kmesh)
    kpts = cell.get_kpts(kmesh)

    scf_obj = pyscf.pbc.scf.KRHF(cell, kpts=kpts)
    scf_obj.exxdiv = None
    dm_kpts = scf_obj.get_init_guess()

    from 
    stdout = sys.stdout if rank == 0 else open(TMPDIR + "/fft_isdf_mpi_%d.log" % rank, "w")
    cell.stdout = stdout

    scf_obj.with_df = ISDF(cell, kpts=kpts)
    scf_obj.with_df.c0 = 10.0
    scf_obj.with_df.verbose = 5
    # scf_obj.with_df.stdout = stdout
    scf_obj.with_df.tol = 1e-10
    scf_obj.with_df.build()

    log = logger.new_logger(None, 5)
    t0 = (process_clock(), perf_counter())
    vj0, vk0 = scf_obj.get_jk(dm_kpts=dm_kpts, with_j=True, with_k=True)
    c0 = scf_obj.with_df.c0
    t1 = log.timer("-> ISDF JK", *t0)
