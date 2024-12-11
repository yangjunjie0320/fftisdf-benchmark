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

PYSCF_MAX_MEMORY = int(os.environ.get("PYSCF_MAX_MEMORY", 160000))

@line_profiler.profile
def build(df_obj, c0=None, m0=None, kpts=None, kmesh=None):
    """
    Build the FFT-ISDF object.
    
    Args:
        df_obj: The FFT-ISDF object to build.
    """
    log = logger.new_logger(df_obj, df_obj.verbose)

    cell = df_obj.cell
    assert numpy.allclose(cell.get_kpts(kmesh), kpts)
    nkpt = len(kpts)

    # build the interpolation vectors
    x_k = df_obj._make_inp_vec(m0=m0, c0=c0, kpts=kpts, kmesh=kmesh)
    nip, nao = x_k.shape[1:]
    assert x_k.shape == (nkpt, nip, nao)
    log.info(
        "Number of interpolation points = %d, effective CISDF = %6.2f",
        nip, nip / nao
    )

    # build the linear equation
    a_k = df_obj._make_lhs(x_k, kpts=kpts, kmesh=kmesh)
    b_k = df_obj._make_rhs(x_k, kpts=kpts, kmesh=kmesh)

    # solve the linear equation
    w_k = df_obj.solve(a_k, b_k, kpts=kpts, kmesh=kmesh)
    assert w_k.shape == (nkpt, nip, nip)

    return x_k, w_k

@line_profiler.profile
def _make_rhs_outcore(df_obj, x_k, kpts=None, kmesh=None, blksize=8000):
    log = logger.new_logger(df_obj, df_obj.verbose)
    t0 = (process_clock(), perf_counter())

    grids = df_obj.grids
    assert grids is not None

    coord = grids.coords
    ngrid = coord.shape[0]

    nkpt = nimg = len(kpts)
    assert numpy.prod(kmesh) == nkpt

    pcell = df_obj.cell
    nao = pcell.nao_nr()
    nip = x_k.shape[1]
    assert x_k.shape == (nkpt, nip, nao)

    wrap_around = df_obj.wrap_around
    scell, phase = get_phase(
        pcell, kpts, kmesh=kmesh,
        wrap_around=wrap_around
    )
    assert phase.shape == (nimg, nkpt)

    log.debug("\nkpt = %d, ngrid = %d, nao = %d", nkpt, ngrid, nao)
    log.debug("ngrid = %d, blksize = %d, nip = %d", ngrid, blksize, nip)
    log.debug("required disk space = %d GB", nkpt * ngrid * nip * 16 / 1e9)

    fswap = df_obj._fswap
    assert fswap is not None

    fswap.create_dataset("rhs", shape=(nkpt, ngrid, nip), dtype=numpy.complex128)
    b_k = fswap["rhs"]
    log.debug("finished creating fswp: %s", fswap.filename)
    
    log.debug("blksize = %d, memory for aoR_loop = %d MB", blksize, blksize * nip * nkpt * 16 / 1e6)
    for ao_k, g0, g1 in df_obj.aoR_loop(grids, kpts, 0, blksize=blksize):
        t_k = numpy.asarray([f @ x.T for f, x in zip(ao_k[0], x_k)])
        assert t_k.shape == (nkpt, g1 - g0, nip)

        t_s = phase @ t_k.reshape(nkpt, -1)
        t_s = t_s.reshape(nimg, g1 - g0, nip)
        t_s = t_s.real

        b_s = (t_s * t_s).reshape(nimg, -1)
        b_k[:, g0:g1, :] = (phase.T @ b_s).reshape(nkpt, g1 - g0, nip)

        log.debug("finished aoR_loop[%8d:%8d]", g0, g1)

    t1 = log.timer("building right-hand side", *t0)
    return b_k

@line_profiler.profile
def _make_lhs_incore(df_obj, x_k, kpts=None, kmesh=None, blksize=8000):
    log = logger.new_logger(df_obj, df_obj.verbose)
    t0 = (process_clock(), perf_counter())

    nkpt = nimg = len(kpts)
    assert numpy.prod(kmesh) == nkpt

    pcell = df_obj.cell
    nao = pcell.nao_nr()
    nip = x_k.shape[1]
    assert x_k.shape == (nkpt, nip, nao)

    wrap_around = df_obj.wrap_around
    scell, phase = get_phase(
        pcell, kpts, kmesh=kmesh,
        wrap_around=wrap_around
    )
    assert phase.shape == (nimg, nkpt)

    nip, nao = x_k.shape[1:]
    assert x_k.shape == (nkpt, nip, nao)

    x2_k = numpy.asarray([x.conj() @ x.T for x in x_k])
    assert x2_k.shape == (nkpt, nip, nip)

    x2_s = phase @ x2_k.reshape(nkpt, -1)
    x2_s = x2_s.reshape(nimg, nip, nip)
    x2_s = x2_s.real

    a_s = x2_s * x2_s
    a_k = phase.conj().T @ (a_s.reshape(nimg, -1))
    a_k = a_k.reshape(nkpt, nip, nip)
    assert a_k.shape == (nkpt, nip, nip)

    t1 = log.timer("building left-hand side", *t0)
    return a_k

@line_profiler.profile
def get_j_kpts(df_obj, dm_kpts, hermi=1, kpts=numpy.zeros((1, 3)), kpts_band=None,
               exxdiv=None):
    """
    Get the J matrix for a set of k-points.
    
    Args:
        df_obj: The FFT-ISDF object. 
        dm_kpts: Density matrices at each k-point.
        hermi: Whether the density matrices are Hermitian.
        kpts: The k-points to calculate J for.
        kpts_band: The k-points of the bands.
        exxdiv: The divergence of the exchange functional (ignored).
        
    Returns:
        The J matrix at the specified k-points.
    """
    cell = df_obj.cell
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1

    pcell = df_obj.cell
    kmesh = df_obj.kmesh
    wrap_around = df_obj.wrap_around
    scell, phase = get_phase(
        pcell, df_obj.kpts, kmesh=kmesh,
        wrap_around=wrap_around
    )

    nao = pcell.nao_nr()
    nkpt = nimg = numpy.prod(kmesh)

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpt, nao = dms.shape[:3]

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    assert nband == nkpt, "not supporting kpts_band"

    x_k = df_obj._x
    w_k = df_obj._w
    w0 = w_k[0].real

    nip = x_k.shape[1]
    assert x_k.shape == (nkpt, nip, nao)
    assert w_k.shape == (nkpt, nip, nip)
    assert w0.shape == (nip, nip)

    rho = numpy.einsum("kIm,kIn,xkmn->xI", x_k, x_k.conj(), dms, optimize=True)
    rho *= 1.0 / nkpt
    assert rho.shape == (nset, nip)

    v0 = numpy.einsum("IJ,xJ->xI", w0, rho, optimize=True)
    assert v0.shape == (nset, nip)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    assert nband == nkpt, "not supporting kpts_band"

    vj_kpts = numpy.einsum("kIm,kIn,xI->xkmn", x_k.conj(), x_k, v0, optimize=True)
    assert vj_kpts.shape == (nset, nkpt, nao, nao)

    if is_zero(kpts_band):
        vj_kpts = vj_kpts.real
    return _format_jks(vj_kpts, dms, input_band, kpts)

@line_profiler.profile
def get_k_kpts(df_obj, dm_kpts, hermi=1, kpts=numpy.zeros((1, 3)), kpts_band=None,
               exxdiv=None):
    cell = df_obj.cell
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1

    pcell = df_obj.cell
    kmesh = df_obj.kmesh
    wrap_around = df_obj.wrap_around
    scell, phase = get_phase(
        pcell, df_obj.kpts, kmesh=kmesh,
        wrap_around=wrap_around
    )

    nao = pcell.nao_nr()
    nkpt = nimg = numpy.prod(kmesh)

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpt, nao = dms.shape[:3]

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    assert nband == nkpt, "not supporting kpts_band"
    assert exxdiv is None, f"exxdiv = {exxdiv}"

    x_k = df_obj._x
    nip = x_k.shape[1]

    w_k = df_obj._w
    w_s = phase @ w_k.reshape(nkpt, -1)
    w_s = w_s.real * numpy.sqrt(nkpt)
    w_s = w_s.reshape(nimg, nip, nip)
    
    assert x_k.shape == (nkpt, nip, nao)
    assert w_k.shape == (nkpt, nip, nip)
    assert w_s.shape == (nimg, nip, nip)

    vk_kpts = []
    for dm in dms:
        rhok = [x @ d @ x.conj().T for x, d in zip(x_k, dm)]
        rhok = numpy.asarray(rhok) / nkpt
        assert rhok.shape == (nkpt, nip, nip)

        rhos = phase @ rhok.reshape(nkpt, -1)
        assert abs(rhos.imag).max() < 1e-10
        rhos = rhos.real.reshape(nimg, nip, nip)

        v_s = w_s * rhos.transpose(0, 2, 1)
        v_s = numpy.asarray(v_s).reshape(nimg, nip, nip)

        v_k = phase.T @ v_s.reshape(nimg, -1)
        v_k = v_k.reshape(nkpt, nip, nip)

        vk_kpts.append([x.conj().T @ v @ x for x, v in zip(df_obj._x, v_k)])

    vk_kpts = numpy.asarray(vk_kpts).reshape(nset, nkpt, nao, nao)
    return _format_jks(vk_kpts, dms, input_band, kpts)

class InterpolativeSeparableDensityFitting(FFTDF):
    """
    Interpolated Separable Density Fitting (ISDF) with FFT
    
    Args:
        cell: The cell object
        kpts: The k-points to use
        kmesh: The k-point mesh to use
        c0: The c0 parameter for ISDF
        m0: The m0 parameter for ISDF
    """
    tol = 1e-10
    blksize = 800
    lstsq_driver = "gelsy"
    wrap_around = False

    _isdf = None
    _x = None
    _w = None
    _fswap = None

    def __init__(self, cell, kpts=numpy.zeros((1, 3)), kmesh=None, c0=0.25, m0=0.0):
        FFTDF.__init__(self, cell, kpts)
        self.kmesh = kmesh
        self.c0 = c0
        self.m0 = m0

        from pyscf.lib import H5TmpFile
        self._fswap = H5TmpFile()
        
    def build(self):
        log = logger.new_logger(self, self.verbose)
        log.info("\n")
        log.info("******** %s ********", self.__class__)
        log.info("method = %s", self.__class__.__name__)

        from pyscf.pbc.tools.k2gamma import kpts_to_kmesh
        kmesh = kpts_to_kmesh(self.cell, self.kpts)
        self.kmesh = kmesh
        log.info("kmesh = %s", kmesh)

        kpts = self.cell.get_kpts(kmesh)
        assert numpy.allclose(self.kpts, kpts), \
            "kpts mismatch, only uniform kpts is supported"

        from pyscf.pbc.tools.pbc import mesh_to_cutoff
        from pyscf.pbc.tools.pbc import cutoff_to_mesh
        m0 = self.m0
        k0 = mesh_to_cutoff(self.cell.a, m0)
        k0 = max(k0)
        log.info("Input parent grid mesh = %s, ke_cutoff = %6.2f", m0, k0)

        m0 = cutoff_to_mesh(self.cell.a, k0)
        c0 = self.c0
        self.m0 = m0
        log.info("Final parent grid size = %s", m0)
        self._x, self._w = build(self, c0=c0, m0=m0, kpts=kpts, kmesh=kmesh)

        if self._isdf is None:
            import tempfile
            isdf = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            self._isdf = isdf.name
        
        log.info("Saving ISDF results to %s", self._isdf)
        from pyscf.lib.chkfile import dump
        dump(self._isdf, "x", self._x)
        dump(self._isdf, "w", self._w)
        return self
    
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

        if kpts is None:
            kpts = self.kpts
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
    
    def _make_inp_vec(self, m0=None, c0=None, kpts=None, kmesh=None):
        if m0 is None:
            m0 = self.m0
        g0 = self.cell.gen_uniform_grids(m0)

        if c0 is None:
            c0 = self.c0

        if kpts is None:
            kpts = self.kpts

        log = logger.new_logger(self, self.verbose)
        t0 = (process_clock(), perf_counter())

        pcell = self.cell
        nao = pcell.nao_nr()
        ng = len(g0)

        nkpt = nimg = len(kpts)

        log.debug("\nSelecting interpolation points")
        log.debug("nkpts = %s, nao = %s, c0 = %6.2f", nkpt, nao, c0)
        log.debug("Parent grid mesh = %s, grid size = %s", m0, ng)

        x2 = numpy.zeros((ng, ng))
        for q in range(nkpt):
            x = pcell.pbc_eval_gto("GTOval", g0, kpts=kpts[q])
            x2 += (x.conj() @ x.T).real
        x4 = (x2 * x2) / nkpt

        from pyscf.lib.scipy_helper import pivoted_cholesky
        tol = self.tol
        chol, perm, rank = pivoted_cholesky(x4, tol=tol)
        if rank == ng:
            log.warn("The parent grid might be too coarse.")

        nip = min(int(nao * c0), rank)
        mask = perm[:nip]

        t1 = log.timer("select interpolation points", *t0)
        log.info(
            "Pivoted Cholesky rank = %d, nip = %d, estimated error = %6.2e",
            rank, nip, chol[nip-1, nip-1]
        )

        x_k = pcell.pbc_eval_gto("GTOval", g0[mask], kpts=kpts)
        x_k = numpy.asarray(x_k, dtype=numpy.complex128)
        return x_k.reshape(nkpt, nip, nao)
    
    def _make_rhs(self, x_k, kpts=None, kmesh=None, blksize=None):
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

        return _make_rhs_outcore(self, x_k, kpts=kpts, kmesh=kmesh, blksize=blksize)
    
    def _make_lhs(self, x_k, kpts=None, kmesh=None):
        if kpts is None:
            kpts = self.kpts

        if kmesh is None:
            kmesh = self.kmesh
            
        return _make_lhs_incore(self, x_k, kpts=kpts, kmesh=kmesh)
    
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
            w_k.append(coul @ z.conj())

            log.info("w[%3d], rank = %4d / %4d", q, rank, a.shape[1])
            t0 = log.timer("w[%3d]" % q, *t0)

        w_k = numpy.asarray(w_k)
        assert w_k.shape == (nkpt, nip, nip)
        return w_k
    
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
    from build import cell_from_poscar

    cell = cell_from_poscar("../data/diamond-conv.vasp")
    cell.basis = 'gth-szv-molopt-sr'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.unit = 'aa'
    cell.precision = 1e-10
    cell.exp_to_discard = 0.1
    cell.max_memory = PYSCF_MAX_MEMORY
    cell.ke_cutoff = 200
    cell.build(dump_input=False)

    kmesh = [2, 2, 2]
    nkpt = nimg = numpy.prod(kmesh)
    kpts = cell.get_kpts(kmesh)

    scf_obj = pyscf.pbc.scf.KRHF(cell, kpts=kpts)
    scf_obj.exxdiv = None
    dm_kpts = scf_obj.get_init_guess()

    scf_obj.with_df = ISDF(cell, kpts=kpts)
    scf_obj.with_df.c0 = 40.0
    scf_obj.with_df.m0 = [15, 15, 15]
    scf_obj.with_df.verbose = 5
    scf_obj.with_df.tol = 1e-20
    scf_obj.with_df.build()

    log = logger.new_logger(None, 5)
    t0 = (process_clock(), perf_counter())
    vj0, vk0 = scf_obj.get_jk(dm_kpts=dm_kpts, with_j=True, with_k=True)
    c0 = scf_obj.with_df.c0
    t1 = log.timer("-> ISDF JK", *t0)

    t0 = (process_clock(), perf_counter())
    scf_obj.with_df = FFTDF(cell, kpts)
    vj1, vk1 = scf_obj.get_jk(dm_kpts=dm_kpts, with_j=True, with_k=True)
    t1 = log.timer("-> FFTDF JK", *t0)

    err = abs(vj0 - vj1).max()
    print("-> ISDF c0 = % 6.4f, vj err = % 6.4e" % (c0, err))

    err = abs(vk0 - vk1).max()
    print("-> ISDF c0 = % 6.4f, vk err = % 6.4e" % (c0, err))
