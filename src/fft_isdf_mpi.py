import mpi4py
from mpi4py import MPI

import os, sys
import numpy, scipy
import scipy.linalg

import pyscf
from pyscf import lib
TMPDIR = lib.param.TMPDIR
from pyscf.lib import logger, current_memory
from pyscf.lib.logger import process_clock, perf_counter
from pyscf.pbc.tools.k2gamma import get_phase

from pyscf.pbc import tools as pbctools
from fft_isdf import InterpolativeSeparableDensityFitting

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print("rank = %d, size = %d" % (rank, size))
comm.Barrier()

class WithMPI(InterpolativeSeparableDensityFitting):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # broadcast the fswap file to all ranks
        fswap_name = None if rank != 0 else self._fswap.filename
        fswap_name = comm.bcast(fswap_name, root=0)


    def _make_inp_vec(self, m0=None, c0=None, kpts=None, kmesh=None):
        if rank == 0:
            res = super()._make_inp_vec(m0, c0, kpts, kmesh)
        else:
            res = None
        res = comm.bcast(res, root=0)
        return res

    def solve(self, a_q, b_q, kpts=None, kmesh=None):
        log = logger.new_logger(self, self.verbose)
        t0 = (process_clock(), perf_counter())

        nkpt = len(kpts)
        nip = a_q.shape[1]

        pcell = self.cell
        nao = pcell.nao_nr()

        wrap_around = self.wrap_around
        scell, phase = get_phase(
            pcell, kpts, kmesh=kmesh,
            wrap_around=wrap_around
        )
        nimg = phase.shape[0]
        assert phase.shape == (nimg, nkpt)

        grids = self.grids
        assert grids is not None
        mesh = grids.mesh
        coord = grids.coords
        ngrid = coord.shape[0]

        assert a_q.shape == (nkpt, nip, nip)
        assert b_q.shape == (nkpt, ngrid, nip)

        w_k = []
        gv = pcell.get_Gv(mesh)

        for q in range(nkpt):
            if q % size != rank:
                continue

            # solving the over-determined linear equation
            # aq @ zq = bq
            a = a_q[q]
            b = b_q[q]
            f = numpy.exp(-1j * numpy.dot(coord, kpts[q]))
            assert f.shape == (ngrid, )

            from scipy.linalg import lstsq
            lstsq_driver = self.lstsq_driver
            tol = self.tol
            res = lstsq(
                a, b.T, cond=tol,
                lapack_driver=lstsq_driver
            )
            z = res[0].T
            rank = res[2]

            zeta = pbctools.fft((f[:, None] * z).T, mesh)
            assert zeta.shape == (nip, ngrid)
            zeta *= pbctools.get_coulG(pcell, k=kpts[q], mesh=mesh, Gv=gv)
            zeta *= pcell.vol / ngrid
            assert zeta.shape == (nip, ngrid)

            from pyscf.pbc.tools.pbc import ifft
            coul = ifft(zeta, mesh) * f.conj()
            w_k.append((q, coul @ z.conj()))

            log.info("w[%3d], rank = %4d / %4d", q, rank, a.shape[1])
            t0 = log.timer("w[%3d]" % q, *t0)

        w_k = comm.gather(w_k, root=0)
        if rank == 0:
            w_k = sorted(w_k, key=lambda x: x[0])
            w_k = numpy.asarray([x[1] for x in w_k])
            assert w_k.shape == (nkpt, nip, nip)
        return w_k
    
    def _make_rhs(self, x_k, kpts=None, kmesh=None):
        if kpts is None:
            kpts = self.kpts
        
        if kmesh is None:
            kmesh = self.kmesh
        
        nip = x_k.shape[1]
        nao = self.cell.nao_nr()
        ngrid = self.grids.coords.shape[0]
        nkpt = len(kpts)

        assert x_k.shape == (nkpt, nip, nao)

        from pyscf.lib import current_memory
        memory = self.max_memory - current_memory()[0]
        blksize = memory / (nkpt * nip * 16) * 1e6 * 0.1
        blksize = max(blksize, self.blksize)
        blksize = int(blksize)

        if rank == 0:
            res = super()._make_rhs(x_k, kpts=kpts, kmesh=kmesh, blksize=blksize)
        else:
            res = None
        return 

ISDF = WithMPI

if __name__ == "__main__":
    from src.utils import cell_from_poscar

    cell = cell_from_poscar("../data/diamond-conv.vasp")
    cell.basis = 'gth-szv-molopt-sr'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.unit = 'aa'
    cell.precision = 1e-10
    cell.exp_to_discard = 0.1
    cell.max_memory = 4000
    cell.ke_cutoff = 20
    cell.build(dump_input=False)

    kmesh = [4, 4, 4]
    nkpt = nimg = numpy.prod(kmesh)
    kpts = cell.get_kpts(kmesh)

    scf_obj = pyscf.pbc.scf.KRHF(cell, kpts=kpts)
    scf_obj.exxdiv = None
    dm_kpts = scf_obj.get_init_guess()

    
    stdout = sys.stdout if rank == 0 else open(TMPDIR + "/fft_isdf_mpi_%d.log" % rank, "w")
    cell.stdout = stdout

    scf_obj.with_df = ISDF(cell, kpts=kpts)
    scf_obj.with_df.c0 = 10.0
    scf_obj.with_df.m0 = [11, 11, 11]
    scf_obj.with_df.verbose = 5
    scf_obj.with_df.stdout = stdout
    scf_obj.with_df.tol = 1e-20
    scf_obj.with_df.build()

    log = logger.new_logger(None, 5)
    t0 = (process_clock(), perf_counter())
    vj0, vk0 = scf_obj.get_jk(dm_kpts=dm_kpts, with_j=True, with_k=True)
    c0 = scf_obj.with_df.c0
    t1 = log.timer("-> ISDF JK", *t0)