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
    """
    Build the FFT-ISDF object.
    
    Args:
        df_obj: The FFT-ISDF object to build.
    """
    log = logger.new_logger(df_obj, df_obj.verbose)
    pcell = df_obj.cell

    kmesh = df_obj.kmesh
    kpts = df_obj.kpts
    nao = pcell.nao_nr()
    nkpt = nimg = numpy.prod(kmesh)

    c0 = df_obj.c0
    m0 = df_obj.m0

    x = df_obj._select(m0=m0, c0=c0, kpts=kpts)
    nip = x.shape[1]
    assert x.shape == (nkpt, nip, nao)
    log.info("Number of interpolation points = %d", nip)

    # build the linear equation
    a = df_obj.make_lhs(x)
    b = df_obj.make_rhs(x)

    # solve the linear equation
    w = scipy.linalg.solve(a, b)
    assert w.shape == (nkpt, nip, nip)

    return x, w

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
    """
    Interpolated Separable Density Fitting (ISDF) with FFT
    
    Args:
        cell: The cell object
        kpts: The k-points to use
        kmesh: The k-point mesh to use
        c0: The c0 parameter for ISDF
        m0: The m0 parameter for ISDF
    """
    blksize = 8000  # block size for the aoR_loop
    lstsq_driver = "gelsy"

    _isdf = None
    _x = None
    _w = None

    def __init__(self, cell, kpts=numpy.zeros((1, 3)), kmesh=None, c0=0.25, m0=0.0):
        FFTDF.__init__(self, cell, kpts)
        self.kmesh = kmesh
        self.c0 = c0
        self.m0 = m0
        
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

        self.m0 = cutoff_to_mesh(self.cell.a, k0)
        log.info("Final parent grid size = %s", self.m0)
        w, x = build(self)

        if self._isdf is None:
            import tempfile
            isdf = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            self._isdf = isdf.name
        
        log.info("Saving ISDF results to %s", self._isdf)
        from pyscf.lib.chkfile import dump
        dump(self._isdf, "x", x)
        dump(self._isdf, "w", w)
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
    
    def get_interpolation_vectors(self, m0=None, c0=None, kpts=None):
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
        nkpt = len(self.kpts)
        ng = len(g0)

        log.debug("\nSelecting interpolation points")
        log.debug("nkpts = %s, nao = %s, c0 = %6.2f", nkpt, nao, c0)
        log.debug("Parent grid mesh = %s, grid size = %s", m0, ng)

        x2 = numpy.zeros((ng, ng), dtype=numpy.double)
        for q, vq in enumerate(self.kpts):
            xq = pcell.pbc_eval_gto("GTOval", g0, kpts=vq)
            x2 += (xq.conj() @ xq.T).real
        x4 = (x2 * x2 / nkpt).real

        from pyscf.lib.scipy_helper import pivoted_cholesky
        chol, perm, rank = pivoted_cholesky(x4)
        if rank == ng:
            log.warn("The parent grid might be too coarse.")

        nip = min(int(nao * c0), rank)
        mask = perm[:nip]

        t1 = log.timer("select interpolation points", *t0)
        log.info(
            "Pivoted Cholesky rank = %d, nip = %d, estimated error = %6.2e",
            rank, nip, chol[nip-1, nip-1]
        )

        xq = pcell.pbc_eval_gto("GTOval", g0[mask], kpts=self.kpts)
        xq = numpy.asarray(xq, dtype=numpy.complex128)
        return xq.reshape(nkpt, nip, nao)
    
    def make_rhs(self, xk):
        log = logger.new_logger(self, self.verbose)
        t0 = (process_clock(), perf_counter())

        grids = self.grids
        assert grids is not None

        coord = grids.coords
        ngrid = coord.shape[0]

        nkpt = len(kpts)
        nao = self.cell.nao_nr()
        nip = xk.shape[1]
        assert xk.shape == (nkpt, nip, nao)

        pcell = self.cell
        kmesh = self.kmesh
        phase, scell = get_phase(pcell, kpts, kmesh=kmesh, wrap_around=False)
        nimg = phase.shape[0]
        assert phase.shape == (nimg, nkpt)

        required_disk_space = nkpt * ngrid * nip * 16 / 1e9
        log.info("nkpt = %d, ngrid = %d, nao = %d, nip = %d", nkpt, ngrid, nao, nip)
        log.info("required disk space = %d GB", required_disk_space)

        from pyscf.lib import H5TmpFile
        fswap = H5TmpFile()
        fswap.create_dataset("y", shape=(nkpt, ngrid, nip), dtype=numpy.complex128)
        y = fswap["y"]
        log.debug("finished creating fswp: %s", fswap.filename)
        
        # compute the memory required for the aoR_loop
        blksize = self.blksize
        required_memory = blksize * nip * nkpt * 16 / 1e6
        log.info("Required memory = %d MB", required_memory)
        
        t0 = (process_clock(), perf_counter())
        for ao_k, g0, g1 in self.aoR_loop(grids, kpts, 0, blksize=blksize):
            t_k = numpy.asarray([f @ x.T for f, x in zip(ao_k[0], xk)])
            assert t_k.shape == (nkpt, g1 - g0, nip)

            t_s = phase @ t_k.reshape(nkpt, -1)
            t_s = t_s.reshape(nimg, g1 - g0, nip)
            t_s = t_s.real

            y_s = t_s * t_s
            y_k = phase.T @ y_s.reshape(nimg, -1)
            y[:, g0:g1, :] = y_k.reshape(nkpt, g1 - g0, nip)

            log.debug("finished aoR_loop[%8d:%8d]", g0, g1)

        return y
    
    def make_lhs(self, xk):
        log = logger.new_logger(self, self.verbose)
        t0 = (process_clock(), perf_counter())

        nkpt = len(kpts)
        nao = self.cell.nao_nr()
        nip = xk.shape[1]
        assert xk.shape == (nkpt, nip, nao)

        pcell = self.cell
        kmesh = self.kmesh
        phase, scell = get_phase(pcell, kpts, kmesh=kmesh, wrap_around=False)
        nimg = phase.shape[0]
        assert phase.shape == (nimg, nkpt)

        nkpt = len(self.kpts)
        nip = xk.shape[1]
        assert xk.shape == (nkpt, nip, nao)

        x2_k = numpy.asarray([x.conj() @ x.T for x in xk])
        assert x2_k.shape == (nkpt, nip, nip)

        x2_s = phase @ x2_k.reshape(nkpt, -1)
        x2_s = x2_s.reshape(nimg, nip, nip)
        x2_s = x2_s.real

        aq = phase.conj().T @ (x2_s * x2_s).reshape(nimg, -1)
        aq = aq.reshape(nkpt, nip, nip)
        assert aq.shape == (nkpt, nip, nip)

        t1 = log.timer("building left-hand side", *t0)
        return aq
    
    def solve(self, aq, bq):
        log = logger.new_logger(self, self.verbose)
        t0 = (process_clock(), perf_counter())

        nkpt = len(kpts)
        nip = aq.shape[1]

        pcell = self.cell
        kmesh = self.kmesh
        phase, scell = get_phase(pcell, kpts, kmesh=kmesh, wrap_around=False)
        nimg = phase.shape[0]
        assert phase.shape == (nimg, nkpt)

        grids = self.grids
        assert grids is not None
        mesh = grids.mesh
        coord = grids.coords
        ngrid = coord.shape[0]

        assert aq.shape == (nkpt, nip, nip)
        assert bq.shape == (nkpt, nip, ngrid)

        wq = []
        gv = pcell.get_Gv(mesh)

        for q in range(nkpt):
            # solving the over-determined linear equation
            # aq @ zq = bq
            a = aq[q]
            b = bq[q]
            f = numpy.exp(-1j * numpy.dot(coord, kpts[q]))
            assert f.shape == (ngrid, )

            from scipy.linalg import lstsq
            lstsq_driver = self.lstsq_driver
            res = lstsq(a, b, lapack_driver=lstsq_driver)
            z = res[0]
            rank = res[2]

            zeta = pbctools.fft(z * f, mesh)
            zeta *= pbctools.get_coulG(pcell, k=kpts[q], mesh=mesh, Gv=gv)
            zeta *= pcell.vol / ngrid
            assert zeta.shape == (nip, ngrid)

            from pyscf.pbc.tools.pbc import ifft
            w = ifft(zeta, mesh) * f.conj()
            wq.append(w @ z.conj().T)

            log.info("finished w[%3d], rank = %4d / %4d", q, rank, a.shape[1])
            t0 = log.timer("w[%3d]", *t0)

        return wq
    
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

    cell = cell_from_poscar("../data/cacuo2-conv-2x2x1.vasp")
    cell.basis = 'gth-szv-molopt-sr'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.unit = 'aa'
    cell.precision = 1e-10
    cell.exp_to_discard = 0.1
    cell.max_memory = PYSCF_MAX_MEMORY
    cell.build(dump_input=False)

    ke_cutoff = cell.ke_cutoff
    print("-> ke_cutoff = %s" % ke_cutoff)

    kmesh = [4, 4, 4]
    nkpt = nimg = numpy.prod(kmesh)
    kpts = cell.get_kpts(kmesh)

    scf_obj = pyscf.pbc.scf.KRHF(cell, kpts=kpts)
    scf_obj.exxdiv = None
    dm_kpts = scf_obj.get_init_guess()

    scf_obj.with_df = ISDF(cell, kpts=kpts)
    scf_obj.with_df.c0 = 40.0
    scf_obj.with_df.m0 = [10, 10, 10]
    scf_obj.with_df.verbose = 5
    scf_obj.with_df.build()

    # vj0, vk0 = scf_obj.get_jk(dm_kpts=dm_kpts, with_j=True, with_k=True)

    # kmesh = [4, 4, 4]
    # nkpt = nimg = numpy.prod(kmesh)
    # kpts = cell.get_kpts(kmesh)

    # log = logger.new_logger(None, 5)
    # scf_obj = pyscf.pbc.scf.KRHF(cell, kpts=cell.get_kpts(kmesh))
    # scf_obj.exxdiv = None
    # dm_kpts = scf_obj.get_init_guess()

    # t0 = (process_clock(), perf_counter())
    # scf_obj.with_df = FFTDF(cell, kpts)
    # vj0, vk0 = scf_obj.get_jk(dm_kpts=dm_kpts, with_j=True, with_k=True)
    # t1 = log.timer("-> FFTDF JK", *t0)

    # from pyscf.pbc.df import GDF
    # scf_obj.with_df = GDF(cell, kpts)
    # scf_obj.with_df.build()
    # t1 = log.timer("-> Building GDF", *t1)
    # vj1, vk1 = scf_obj.get_jk(dm_kpts=dm_kpts, with_j=True, with_k=True)
    # t1 = log.timer("-> GDF JK", *t1)

    # err = abs(vj0 - vj1).max()
    # print("-> GDF vj err = % 6.4e" % err)

    # err = abs(vk0 - vk1).max()
    # print("-> GDF vk err = % 6.4e" % err)

    # scf_obj.with_df = ISDF(cell, kpts=cell.get_kpts(kmesh))
    # scf_obj.with_df.verbose = 5
    # scf_obj.with_df.c0 = 40.0
    # scf_obj.with_df.m0 = [15, 15, 15]
    # scf_obj.with_df.build()
    # t1 = log.timer("-> Building ISDF", *t1)
    # vj1, vk1 = scf_obj.get_jk(dm_kpts=dm_kpts, with_j=True, with_k=True)
    # t1 = log.timer("-> ISDF JK", *t1)

    # c0 = scf_obj.with_df.c0
    # err = abs(vj0 - vj1).max()
    # print("-> ISDF c0 = % 6.4f, vj err = % 6.4e" % (c0, err))

    # err = abs(vk0 - vk1).max()
    # print("-> ISDF c0 = % 6.4f, vk err = % 6.4e" % (c0, err))
