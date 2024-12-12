import itertools, os, sys
from itertools import product

import numpy, scipy
from opt_einsum import contract as einsum
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

PYSCF_MAX_MEMORY = int(os.environ.get("PYSCF_MAX_MEMORY", 160000))

def build(df_obj):
    log = logger.new_logger(df_obj, df_obj.verbose)
    pcell = df_obj.cell

    kmesh = df_obj.kmesh
    vk = df_obj.kpts
    scell, phase = get_phase(pcell, vk, kmesh=kmesh, wrap_around=False)

    nao = pcell.nao_nr()
    nkpt = nimg = numpy.prod(kmesh)

    xip = df_obj.select_interpolation_points()
    nip = xip.shape[1]
    assert xip.shape == (nkpt, nip, nao)
    log.info("Number of interpolation points = %d", nip)

    x2_k = numpy.asarray([xq.conj() @ xq.T for xq in xip])
    assert x2_k.shape == (nkpt, nip, nip)

    x2_s = phase @ x2_k.reshape(nkpt, -1)
    x2_s = x2_s.reshape(nimg, nip, nip)
    assert abs(x2_s.imag).max() < 1e-10

    x4_s = x2_s * x2_s
    x4_k = phase.conj().T @ x4_s.reshape(nimg, -1)
    x4_k = x4_k.reshape(nkpt, nip, nip)
    assert x4_k.shape == (nkpt, nip, nip)

    t0 = (process_clock(), perf_counter())

    grids = df_obj.grids
    coord = grids.coords
    ngrid = coord.shape[0]

    required_disk_space = nkpt * ngrid * nip * 16 / 1e9
    log.info("nkpt = %d, ngrid = %d, nip = %d", nkpt, ngrid, nip)
    log.info("Required disk space = %d GB", required_disk_space)

    from pyscf.lib import H5TmpFile
    fswap = H5TmpFile()
    fswap.create_dataset("y", shape=(nkpt, ngrid, nip), dtype=numpy.complex128)
    y = fswap["y"]
    log.debug("finished creating fswp: %s", fswap.filename)
    
    # compute the memory required for the aoR_loop
    blksize = df_obj.blksize
    required_memory = blksize * nip * nkpt * 16 / 1e6
    log.info("Required memory = %d MB", required_memory)
    
    t0 = (process_clock(), perf_counter())
    for ao_k_etc, g0, g1 in df_obj.aoR_loop(grids, vk, 0, blksize=blksize):
        f_k = numpy.asarray(ao_k_etc[0])
        assert f_k.shape == (nkpt, g1 - g0, nao)

        fx_k = numpy.asarray([f.conj() @ x.T for f, x in zip(f_k, xip)])
        assert fx_k.shape == (nkpt, g1 - g0, nip)

        fx_s = phase @ fx_k.reshape(nkpt, -1)
        fx_s = fx_s.reshape(nimg, g1 - g0, nip)
        assert abs(fx_s.imag).max() < 1e-10

        y_s = fx_s * fx_s
        y_k = phase.T @ y_s.reshape(nimg, -1)
        y[:, g0:g1, :] = y_k.reshape(nkpt, g1 - g0, nip)

        log.debug("finished aoR_loop[%8d:%8d]", g0, g1)

    t1 = log.timer("building y", *t0)
    mesh = df_obj.mesh
    gv = pcell.get_Gv(mesh)
    
    required_memory = nip * ngrid * 16 / 1e6
    log.info("Required memory = %d MB", required_memory)

    wq = []
    for q, vq in enumerate(vk):
        t0 = (process_clock(), perf_counter())
        fq = numpy.exp(-1j * numpy.dot(coord, vq))
        assert fq.shape == (ngrid, )
        
        y_q = y[q, :, :]
        assert y_q.shape == (ngrid, nip)

        x4_q = x4_k[q]
        assert x4_q.shape == (nip, nip)

        res = scipy.linalg.lstsq(x4_q, y_q.T, lapack_driver="gelsy")
        z_q = res[0]
        rank = res[2]
        assert z_q.shape == (nip, ngrid)
        
        zeta_q = pbctools.fft(z_q * fq, mesh)
        zeta_q *= pbctools.get_coulG(pcell, k=vq, mesh=mesh, Gv=gv)
        zeta_q *= pcell.vol / ngrid
        assert zeta_q.shape == (nip, ngrid)

        zeta_q = pbctools.ifft(zeta_q, mesh)
        zeta_q *= fq.conj()

        wq.append(zeta_q @ z_q.conj().T)
        t1 = log.timer("w[%3d], rank = %4d / %4d" % (q, rank, nip), *t0)

    wq = numpy.asarray(wq).reshape(nkpt, nip, nip)
    df_obj._x = xip

    df_obj._w0 = wq[0]
    df_obj._wq = wq

def get_j_kpts(df_obj, dm_kpts, hermi=1, kpts=numpy.zeros((1, 3)), kpts_band=None,
               exxdiv=None):
    log = logger.new_logger(df_obj, df_obj.verbose)
    cell = df_obj.cell
    mesh = df_obj.mesh
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1
    pcell = df_obj.cell

    assert exxdiv is None
    kpts = numpy.asarray(kpts)
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpt, nao = dms.shape[:3]

    assert df_obj._x is not None
    assert df_obj._w0 is not None

    nip = df_obj._x.shape[1]
    assert df_obj._x.shape  == (nkpt, nip, nao)
    assert df_obj._w0.shape == (nip, nip)

    rho = numpy.einsum("kIm,kIn,xkmn->xI", df_obj._x, df_obj._x.conj(), dms, optimize=True)
    rho *= 1.0 / nkpt
    assert rho.shape == (nset, nip)

    v = numpy.einsum("IJ,xJ->xI", df_obj._w0, rho, optimize=True)
    assert v.shape == (nset, nip)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    assert nband == nkpt

    vj_kpts = numpy.einsum("kIm,kIn,xI->xkmn", df_obj._x.conj(), df_obj._x, v, optimize=True)
    assert vj_kpts.shape == (nset, nkpt, nao, nao)

    if is_zero(kpts_band):
        vj_kpts = vj_kpts.real
    return _format_jks(vj_kpts, dms, input_band, kpts)

def get_k_kpts(df_obj, dm_kpts, hermi=1, kpts=numpy.zeros((1, 3)), kpts_band=None,
               exxdiv=None):
    log = logger.new_logger(df_obj, df_obj.verbose)
    cell = df_obj.cell
    mesh = df_obj.mesh
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1

    pcell = df_obj.cell
    kmesh = df_obj.kmesh
    scell, phase = get_phase(pcell, df_obj.kpts, kmesh=kmesh, wrap_around=False)

    nao = pcell.nao_nr()
    nkpt = nimg = numpy.prod(kmesh)

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpt, nao = dms.shape[:3]

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    assert nband == nkpt # not supporting kpts_band
    assert exxdiv is None

    assert df_obj._x is not None
    assert df_obj._wq is not None

    nip = df_obj._x.shape[1]
    assert df_obj._x.shape == (nkpt, nip, nao)
    assert df_obj._wq.shape == (nkpt, nip, nip)

    wq = df_obj._wq
    ws = phase @ wq.reshape(nkpt, -1)
    ws = ws.reshape(nimg, nip, nip)
    ws = ws.real * numpy.sqrt(nkpt)

    vk_kpts = []
    for dm in dms:
        rhok = [x @ d @ x.conj().T for x, d in zip(df_obj._x, dm)]
        rhok = numpy.asarray(rhok) / nkpt
        assert rhok.shape == (nkpt, nip, nip)

        rhos = phase @ rhok.reshape(nkpt, -1)
        assert abs(rhos.imag).max() < 1e-10
        rhos = rhos.real.reshape(nimg, nip, nip)

        vs = ws * rhos.transpose(0, 2, 1)
        vs = numpy.asarray(vs).reshape(nimg, nip, nip)

        vk = phase.T @ vs.reshape(nimg, -1)
        vk = vk.reshape(nkpt, nip, nip)

        vk_kpts.append([x.conj().T @ v @ x for x, v in zip(df_obj._x, vk)])

    vk_kpts = numpy.asarray(vk_kpts).reshape(nset, nkpt, nao, nao)
    return _format_jks(vk_kpts, dms, input_band, kpts)

class InterpolativeSeparableDensityFitting(FFTDF):
    _x = None
    _w0 = None
    _wq = None
    blksize = 8000  # block size for the aoR_loop

    def __init__(self, cell, kpts, m0=None, c0=20.0):
        super().__init__(cell, kpts)

        self.m0 = m0 if m0 is not None else [15, 15, 15]
        self.c0 = c0

    def build(self):
        log = logger.new_logger(self, self.verbose)
        log.info("\n")
        log.info("******** %s ********", self.__class__)
        log.info("method = %s", self.__class__.__name__)

        log.info("Transform kpts to kmesh")
        log.info("original    kpts  =\n %s", self.kpts)
        
        from pyscf.pbc.tools.k2gamma import kpts_to_kmesh
        kmesh = kpts_to_kmesh(self.cell, self.kpts)
        self.kmesh = kmesh
        log.info("transformed kmesh = %s", kmesh)

        self.kpts = cell.get_kpts(kmesh)
        log.info("transformed kpts =\n %s", self.kpts)

        return build(self)
    
    def aoR_loop(self, grids=None, kpts=None, deriv=0, blksize=None):
        if grids is None:
            grids = self.grids
            cell = self.cell
        else:
            cell = grids.cell

        if grids.non0tab is None:
            grids.build(with_non0tab=True)

        if blksize is None:
            blksize = self.blksize

        if kpts is None: kpts = self.kpts
        kpts = numpy.asarray(kpts)

        assert cell.dimension == 3

        max_memory = max(2000, self.max_memory - current_memory()[0])

        ni = self._numint
        nao = cell.nao_nr()
        p1 = 0
        for ao_k1_etc in ni.block_loop(cell, grids, nao, deriv, kpts,
                                       max_memory=max_memory,
                                       blksize=blksize):
            coords = ao_k1_etc[4]
            p0, p1 = p1, p1 + coords.shape[0]
            yield ao_k1_etc, p0, p1
    
    def select_interpolation_points(self, x0=None, phase=None):
        c0 = self.c0
        m0 = self.m0

        log = logger.new_logger(self, self.verbose)
        t0 = (process_clock(), perf_counter())
        # the primitive cell
        pcell = self.cell
        nao = pcell.nao_nr()

        x0 = pcell.pbc_eval_gto(
            "GTOval", pcell.gen_uniform_grids(m0), 
            kpts=self.kpts
        )
        x0 = numpy.asarray(x0)

        nkpt, ng = x0.shape[:2]
        assert x0.shape == (nkpt, ng, nao)

        x2 = numpy.zeros((ng, ng), dtype=numpy.double)
        for q in range(nkpt):
            x2 += (x0[q].conj() @ x0[q].T).real
        x4 = (x2 * x2 / nkpt).real

        from pyscf.lib.scipy_helper import pivoted_cholesky
        chol, perm, rank = pivoted_cholesky(x4)
        nip = min(int(nao * c0), rank)
        mask = perm[:nip]

        t1 = log.timer("select_interpolation_points", *t0)
        log.info("Pivoted Cholesky rank = %d, nip = %d, estimated error = %6.2e", rank, nip, chol[nip, nip])
        return x0[:, mask, :]
    
    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None:
            raise NotImplementedError
        
        if exxdiv is not None:
            raise NotImplementedError

        from pyscf.pbc.df.aft import _check_kpts
        kpts, is_single_kpt = _check_kpts(self, kpts)
        if is_single_kpt:
            raise NotImplementedError
        else:
            vj = vk = None
            if with_k:
                vk = get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
            if with_j:
                vj = get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk
    
ISDF = InterpolativeSeparableDensityFitting

if __name__ == "__main__":
    DATA_PATH = os.getenv("DATA_PATH", None)
    from build import cell_from_poscar

    cell = cell_from_poscar(os.path.join(DATA_PATH, "diamond-prim.vasp"))
    cell.basis = 'gth-dzvp-molopt-sr'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.unit = 'aa'
    cell.exp_to_discard = 0.1
    cell.max_memory = PYSCF_MAX_MEMORY
    cell.mesh = [5, 5, 5]
    cell.build(dump_input=False)

    kmesh = [4, 4, 4]
    nkpt = nimg = numpy.prod(kmesh)
    kpts = cell.get_kpts(kmesh)

    scf_obj = pyscf.pbc.scf.KRHF(cell, kpts=kpts)
    scf_obj.exxdiv = None
    dm_kpts = scf_obj.get_init_guess(key="1e")

    scf_obj.with_df = ISDF(cell, kpts=kpts)
    scf_obj.with_df.c0 = 60.0
    scf_obj.with_df.m0 = [15, 15, 15]
    scf_obj.with_df.verbose = 5
    scf_obj.with_df.tol = 1e-10
    scf_obj.with_df.build()

    log = logger.new_logger(None, 5)
    t0 = (process_clock(), perf_counter())
    vj0, vk0 = scf_obj.get_jk(dm_kpts=dm_kpts, with_j=True, with_k=True)
    c0 = scf_obj.with_df.c0
    t1 = log.timer("-> ISDF JK", *t0)

    t0 = (process_clock(), perf_counter())
    scf_obj.with_df = FFTDF(cell, kpts)
    scf_obj.with_df.verbose = 200
    scf_obj.with_df.dump_flags()
    scf_obj.with_df.check_sanity()
    scf_obj.with_df.mesh = [5, 5, 5]
    vj1, vk1 = scf_obj.get_jk(dm_kpts=dm_kpts, with_j=True, with_k=True)
    t1 = log.timer("-> FFTDF JK", *t0)

    err = abs(vj0 - vj1).max()
    print("-> ISDF c0 = % 6.4f, vj err = % 6.4e" % (c0, err))

    err = abs(vk0 - vk1).max()
    print("-> ISDF c0 = % 6.4f, vk err = % 6.4e" % (c0, err))
