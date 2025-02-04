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

# Naming convention:
# *_kpt: k-space array, which shapes as (nkpt, x, x)
# *_spc: super-cell stripe array, which shapes as (nspc, x, x)
# *_full: full array, shapes as (nspc * x, nspc * x)
# *_k1, *_k2: the k-space array at specified k-point

def spc_to_kpt(m_spc, phase):
    """Convert a matrix from the stripe form (in super-cell)
    to the k-space form.
    """
    nspc, nkpt = phase.shape
    m_kpt = lib.dot(phase.conj().T, m_spc.reshape(nspc, -1))
    return m_kpt.reshape(m_spc.shape)

def kpt_to_spc(m_kpt, phase):
    """Convert a matrix from the k-space form to
    stripe form (in super-cell).
    """
    nspc, nkpt = phase.shape
    m_spc = lib.dot(phase, m_kpt.reshape(nkpt, -1))
    return m_spc.reshape(m_kpt.shape)

def lstsq(a, b, tol=1e-10):
    """
    Solve the least squares problem of the form:
        x = ainv @ b @ ainv.conj().T
    using SVD. In which a is not full rank, and
    ainv is the pseudo-inverse of a.

    Args:
        a: The matrix A.
        b: The matrix B.
        tol: The tolerance for the singular values.

    Returns:
        x: The solution to the least squares problem.
        rank: The rank of the matrix a.
    """

    # make sure a is Hermitian
    assert numpy.allclose(a, a.conj().T)

    u, s, vh = scipy.linalg.svd(a, full_matrices=False)
    uh = u.conj().T
    v = vh.conj().T

    r = s[None, :] * s[:, None]
    m = abs(r) > tol
    rank = m.sum() / m.shape[0]
    t = (uh @ b @ u) * m / r
    return v @ t @ vh, int(rank)

@line_profiler.profile
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
        t0 = (process_clock(), perf_counter())
        from pyscf.lib import H5TmpFile
        fswp = H5TmpFile()

        # metx_q: metric for least-squares
        # eta_q: right-hand side for least-squares
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

        kern_q = get_coul(
            df_obj, eta_q, kpt=kpts[q], 
            tol=tol, fswp=fswp
        )
        coul_q, rank = lstsq(metx_q, kern_q, tol=tol)
        assert coul_q.shape == (nip, nip)

        coul_kpt.append(coul_q)
        log.timer("solving Coulomb kernel", *t0)
        log.info("Finished solving Coulomb kernel for q = %3d / %3d, rank = %d / %d", q + 1, nkpt, rank, nip)

    coul_kpt = numpy.asarray(coul_kpt)
    return inpv_kpt, coul_kpt

@line_profiler.profile
def get_lhs_and_rhs(df_obj, inpv_kpt, kpt=None, blksize=8000, fswp=None):
    log = logger.new_logger(df_obj, df_obj.verbose)
    t0 = (process_clock(), perf_counter())

    grids = df_obj.grids
    assert grids is not None

    coord = grids.coords
    ngrid = coord.shape[0]

    from pyscf.pbc.tools.k2gamma import kpts_to_kmesh
    kpts = df_obj.kpts
    kmesh = df_obj.kmesh
    nkpt = nspc = len(kpts)
    assert numpy.prod(kmesh) == nkpt

    ix = numpy.where(numpy.linalg.norm(kpts - kpt, axis=1) < 1e-10)[0]
    assert len(ix) == 1
    q = ix[0]
    assert numpy.allclose(kpts[q], kpt)

    pcell = df_obj.cell
    nao = pcell.nao_nr()
    nip = inpv_kpt.shape[1]
    assert inpv_kpt.shape == (nkpt, nip, nao)

    wrap_around = df_obj.wrap_around
    scell, phase = get_phase(
        pcell, kpts, kmesh=kmesh,
        wrap_around=wrap_around
    )
    assert phase.shape == (nspc, nkpt)

    log.debug("\nnkpt = %d, nao = %d", nkpt, nao)
    log.debug("ngrid = %d, blksize = %d", ngrid, blksize)
    log.debug("required disk space = %d GB", ngrid * nip * 16 / 1e9)

    t_kpt = numpy.asarray([xk.conj() @ xk.T for xk in inpv_kpt])
    assert t_kpt.shape == (nkpt, nip, nip)

    t_spc = kpt_to_spc(t_kpt, phase)
    assert t_spc.shape == (nspc, nip, nip)

    aq = numpy.zeros((nip, nip), dtype=numpy.complex128)
    for s, ts in enumerate(t_spc):
        aq += phase[s, q] * ts * ts

    bq = None
    if fswp is not None:
        bq = fswp.create_dataset("bq", data=numpy.zeros((ngrid, nip), dtype=numpy.complex128))
        log.debug("Saving bq to %s, memory for bq = %d GB", fswp.filename, bq.size * 16 / 1e9)
    else:
        bq = numpy.zeros((ngrid, nip), dtype=numpy.complex128)
        log.debug("Memory for bq = %d GB", bq.size * 16 / 1e9)
    assert bq is not None

    log.debug("blksize = %d, memory for aoR_loop = %d MB", blksize, blksize * nip * nkpt * 16 / 1e6)
    for ao_kpt, g0, g1 in df_obj.aoR_loop(grids, kpts, 0, blksize=blksize):
        t_kpt = numpy.asarray([fk.conj() @ xk.T for fk, xk in zip(ao_kpt[0], inpv_kpt)])
        assert t_kpt.shape == (nkpt, g1 - g0, nip) # this term scale as O(nkpt * nip * nao * ng)

        t_spc = kpt_to_spc(t_kpt, phase)
        t_spc = t_spc.reshape(nspc, g1 - g0, nip)

        for s, ts in enumerate(t_spc):
            bq[g0:g1] += phase[s, q] * ts * ts

    log.timer("get_lhs_and_rhs", *t0)
    return aq, bq

def get_coul(df_obj, eta_q, kpt=None, tol=1e-10, fswp=None):
    log = logger.new_logger(df_obj, df_obj.verbose)
    t0 = (process_clock(), perf_counter())

    ngrid, nip = eta_q.shape
    assert eta_q.shape == (ngrid, nip)

    log = logger.new_logger(df_obj, df_obj.verbose)
    t0 = (process_clock(), perf_counter())

    nkpt = len(kpts)
    pcell = df_obj.cell
    nao = pcell.nao_nr()

    wrap_around = df_obj.wrap_around
    scell, phase = get_phase(
        pcell, kpts, kmesh=kmesh,
        wrap_around=wrap_around
    )
    nspc = phase.shape[0]
    assert phase.shape == (nspc, nkpt)

    grids = df_obj.grids
    assert grids is not None
    mesh = grids.mesh
    coord = grids.coords
    ngrid = coord.shape[0]

    # assume (nip, nip) could be maintained in memory
    # while (nip, ngrid) could not
    kern_q = numpy.zeros((nip, nip), dtype=numpy.complex128)

    gv = pcell.get_Gv(mesh)
    max_memory = max(2000, df_obj.max_memory - current_memory()[0]) * 0.1
    blksize = max(max_memory * 1e6 // (ngrid * 16), 1)
    blksize = min(int(blksize), nip)

    for i0, i1 in lib.prange(0, nip, blksize):
        eta_qi = eta_q[:, i0:i1]
        assert eta_qi.shape == (ngrid, i1 - i0)

        t = numpy.dot(coord, kpt)
        f = numpy.exp(-1j * t)
        assert f.shape == (ngrid, )

        v_qi = pbctools.fft(eta_qi.T * f, mesh)
        v_qi *= pbctools.get_coulG(pcell, k=kpt, mesh=mesh, Gv=gv)
        v_qi *= pcell.vol / ngrid

        from pyscf.pbc.tools.pbc import ifft
        w_qi = ifft(v_qi, mesh) * f.conj()
        w_qi = w_qi.T
        assert w_qi.shape == (ngrid, i1 - i0)

        for j0, j1 in lib.prange(0, nip, blksize):
            kern_qij = numpy.dot(w_qi.T.conj(), eta_q[:, j0:j1])
            assert kern_qij.shape == (i1 - i0, j1 - j0)
            kern_q[i0:i1, j0:j1] = kern_qij
            kern_q[j0:j1, i0:i1] = kern_qij.conj()

    return kern_q

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
    nkpt = nspc = numpy.prod(kmesh)

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpt, nao = dms.shape[:3]

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    assert nband == nkpt, "not supporting kpts_band"

    inpv_kpt = df_obj._inpv_kpt
    coul_kpt = df_obj._coul_kpt

    nip = inpv_kpt.shape[1]
    assert inpv_kpt.shape == (nkpt, nip, nao)
    assert coul_kpt.shape == (nkpt, nip, nip)

    coul0 = coul_kpt[0]
    assert coul0.shape == (nip, nip)

    rho = numpy.einsum("kIm,kIn,xkmn->xI", inpv_kpt, inpv_kpt.conj(), dms, optimize=True)
    rho *= 1.0 / nkpt
    assert rho.shape == (nset, nip)

    v0 = numpy.einsum("IJ,xJ->xI", coul0, rho, optimize=True)
    assert v0.shape == (nset, nip)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    assert nband == nkpt, "not supporting kpts_band"

    vj_kpts = numpy.einsum("kIm,kIn,xI->xkmn", inpv_kpt.conj(), inpv_kpt, v0, optimize=True)
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
    nkpt = nspc = numpy.prod(kmesh)

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpt, nao = dms.shape[:3]

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    assert nband == nkpt, "not supporting kpts_band"
    assert exxdiv is None, f"exxdiv = {exxdiv}"

    inpv_kpt = df_obj._inpv_kpt
    coul_kpt = df_obj._coul_kpt

    nip = inpv_kpt.shape[1]
    coul_spc = kpt_to_spc(coul_kpt, phase)
    coul_spc = coul_spc * numpy.sqrt(nkpt)
    coul_spc = coul_spc.reshape(nspc, nip, nip)
    
    assert inpv_kpt.shape == (nkpt, nip, nao)
    assert coul_kpt.shape == (nkpt, nip, nip)
    assert coul_spc.shape == (nspc, nip, nip)

    vks = []
    for dm_kpt in dms:
        rho_kpt = [x @ d @ x.conj().T for x, d in zip(inpv_kpt, dm_kpt)]
        rho_kpt = numpy.asarray(rho_kpt) / nkpt
        assert rho_kpt.shape == (nkpt, nip, nip)

        rho_spc = kpt_to_spc(rho_kpt, phase)
        assert rho_spc.shape == (nspc, nip, nip)
        rho_spc = rho_spc.transpose(0, 2, 1)

        v_spc = coul_spc * rho_spc
        v_spc = numpy.asarray(v_spc).reshape(nspc, nip, nip)

        v_kpt = phase.T.conj() @ v_spc.reshape(nspc, -1)
        v_kpt = v_kpt.reshape(nkpt, nip, nip)
        assert v_kpt.shape == (nkpt, nip, nip)

        vks.append([xk.conj().T @ vk @ xk for xk, vk in zip(inpv_kpt, v_kpt)])

    vks = numpy.asarray(vks).reshape(nset, nkpt, nao, nao)
    return _format_jks(vks, dms, input_band, kpts)

class InterpolativeSeparableDensityFitting(FFTDF):
    wrap_around = False

    _isdf = None
    _isdf_to_save = None

    _x = None
    _w = None
    _fswap = None

    _keys = ['_isdf', '_coul_kpt', '_inpv_kpt']

    def __init__(self, cell, kpts=numpy.zeros((1, 3)), kmesh=None, c0=20.0):
        FFTDF.__init__(self, cell, kpts)

        # from pyscf.pbc.lib.kpts import KPoints
        # self.kpts = KPoints(cell, kpts)
        # self.kpts.build()

        self.kmesh = kmesh
        self.c0 = c0

        self.tol = 1e-10
        self.blksize = 800

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("\n")
        log.info("******** %s ********", self.__class__)
        log.info("mesh = %s (%d PWs)", self.mesh, numpy.prod(self.mesh))
        log.info("len(kpts) = %d", len(self.kpts))
        log.debug1("    kpts = %s", self.kpts)
        return self
        
    def build(self):
        self.dump_flags()
        self.check_sanity()

        from pyscf.pbc.tools.k2gamma import kpts_to_kmesh
        kmesh = kpts_to_kmesh(self.cell, self.kpts)
        self.kmesh = kmesh

        log = logger.new_logger(self, self.verbose)
        log.info("kmesh = %s", kmesh)
        t0 = (process_clock(), perf_counter())

        kpts = self.cell.get_kpts(kmesh)
        assert numpy.allclose(self.kpts, kpts), \
            "kpts mismatch, only uniform kpts is supported"

        if self._isdf is not None:
            raise NotImplementedError

        inpv_kpt, coul_kpt = build(
            df_obj=self,
            c0=self.c0,
            kpts=kpts,
            kmesh=kmesh
        )

        self._coul_kpt = coul_kpt
        self._inpv_kpt = inpv_kpt

        if self._isdf_to_save is not None:
            self._isdf = self._isdf_to_save

        if self._isdf is None:
            import tempfile
            isdf = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            self._isdf = isdf.name
        
        log.info("Saving FFTISDF results to %s", self._isdf)
        from pyscf.lib.chkfile import dump
        dump(self._isdf, "coul_kpt", coul_kpt)
        dump(self._isdf, "inpv_kpt", inpv_kpt)

        t1 = log.timer("building ISDF", *t0)
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

        block_loop = ni.block_loop(
            cell, grids, nao, deriv, kpts,
            max_memory=max_memory,
            blksize=blksize
            )
        
        for ao_etc_kpt in block_loop:
            coords = ao_etc_kpt[4]
            p0, p1 = p1, p1 + coords.shape[0]
            yield ao_etc_kpt, p0, p1
    
    def get_inpv(self, c0=None):
        nao = self.cell.nao_nr()
        nip = nao * c0

        nkpt = len(self.kpts)

        from pyscf.pbc.tools.pbc import mesh_to_cutoff
        lv = self.cell.lattice_vectors()
        k0 = mesh_to_cutoff(lv, [int(numpy.power(nip, 1/3) + 1)] * 3)
        k0 = max(k0)

        from pyscf.pbc.tools.pbc import cutoff_to_mesh
        g0 = self.cell.gen_uniform_grids(cutoff_to_mesh(lv, k0))
        log = logger.new_logger(self, self.verbose)

        pcell = self.cell
        nao = pcell.nao_nr()
        ng = len(g0)

        # x2 = numpy.zeros((ng, ng))
        # x2 = pcell.pbc_eval_gto("GTOval", g0)
        # x2 = (x2.conj() @ x2.T).real
        # x4 = (x2 * x2) / nkpt
        t0 = numpy.zeros((ng, ng))
        for q in range(nkpt):
            xq = pcell.pbc_eval_gto("GTOval", g0, kpts=self.kpts[q])
            tq = numpy.dot(xq.conj(), xq.T)
            t0 += tq.real
        m0 = (t0 * t0) / nkpt

        from pyscf.lib.scipy_helper import pivoted_cholesky
        tol = self.tol
        chol, perm, rank = pivoted_cholesky(m0, tol=tol ** 2)
        if rank == ng:
            log.warn("The parent grid might be too coarse.")

        nip = min(int(nip), rank)
        mask = perm[:nip]

        log.info(
            "Pivoted Cholesky rank = %d, nip = %d, estimated error = %6.2e",
            rank, nip, chol[nip-1, nip-1]
        )

        return g0[mask]
    
    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        assert omega is None and exxdiv is None

        from pyscf.pbc.df.aft import _check_kpts
        kpts, is_single_kpt = _check_kpts(self, kpts)
        if is_single_kpt:
            raise NotImplementedError

        vj = vk = None
        if with_k:
            vk = get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            vj = get_j_kpts(self, dm, hermi, kpts, kpts_band)

        return vj, vk

ISDF = FFTISDF = InterpolativeSeparableDensityFitting

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

    # kmesh = [4, 4, 4]
    kmesh = [2, 2, 2]
    nkpt = nspc = numpy.prod(kmesh)
    kpts = cell.get_kpts(kmesh)

    scf_obj = pyscf.pbc.scf.KRHF(cell, kpts=kpts)
    scf_obj.exxdiv = None
    dm_kpts = scf_obj.get_init_guess(key="minao")

    log = logger.new_logger(None, 5)

    t0 = (process_clock(), perf_counter())
    scf_obj.with_df = FFTDF(cell, kpts)
    scf_obj.with_df.verbose = 5
    scf_obj.with_df.dump_flags()
    scf_obj.with_df.check_sanity()

    vj1 = numpy.zeros((nkpt, nao, nao))
    vk1 = numpy.zeros((nkpt, nao, nao))
    vj0, vk0 = scf_obj.get_jk(dm_kpts=dm_kpts, with_j=True, with_k=True)
    vj1 = vj1.reshape(nkpt, nao, nao)
    vk1 = vk1.reshape(nkpt, nao, nao)
    t1 = log.timer("-> FFTDF JK", *t0)

    for c0 in [5.0, 10.0, 15.0, 20.0]:
        t0 = (process_clock(), perf_counter())
        # c0 = 40.0
        scf_obj.with_df = ISDF(cell, kpts=kpts)
        scf_obj.with_df.c0 = c0
        scf_obj.with_df.verbose = 0
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
        t1 = log.timer("-> ISDF JK", *t0)

        err = abs(vj0 - vj1).max()
        print("-> ISDF c0 = % 6.2f, vj err = % 6.4e" % (c0, err))

        err = abs(vk0 - vk1).max()
        print("-> ISDF c0 = % 6.2f, vk err = % 6.4e" % (c0, err))

    # from pyscf.pbc.tools.pbc import cutoff_to_mesh
    # m0 = cutoff_to_mesh(cell.a, 40.0)
    # x_k = df_obj._make_inp_vec(m0=m0, c0=c0, kpts=kpts, kmesh=kmesh)
    # nip, nao = x_k.shape[1:]
    # assert x_k.shape == (nkpt, nip, nao)

    # b_k = df_obj._make_rhs(x_k, kpts=kpts, kmesh=kmesh)

    # for q in range(nkpt):
    #     b_q_sol = _make_rhs_incore(df_obj, x_k, q=q, kpts=kpts, kmesh=kmesh, blksize=1000)
    #     b_q_ref = b_k[q]

    #     err = abs(b_q_ref - b_q_sol).max()

    #     if err > 1e-8:
    #         from sys import stdout
            
    #         print("err = %6.4e, q = %d", err, q)
    #         print(f"b_q_ref real = ")
    #         numpy.savetxt(stdout, b_q_ref.real[:10, :10], delimiter=", ", fmt="% 6.2e")

    #         print("b_q_ref imag = ")
    #         numpy.savetxt(stdout, b_q_ref.imag[:10, :10], delimiter=", ", fmt="% 6.2e")

    #         print("b_q_sol real = ")
    #         numpy.savetxt(stdout, b_q_sol.real[:10, :10], delimiter=", ", fmt="% 6.2e")

    #         print("b_q_sol imag = ")
    #         numpy.savetxt(stdout, b_q_sol.imag[:10, :10], delimiter=", ", fmt="% 6.2e")

    #         assert 1 == 2
