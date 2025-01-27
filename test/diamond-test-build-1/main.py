import pyscf.isdf.BackEnd._config as config

config.disable_fftw()
# config.backend("numpy")
# config.backend("scipy")
config.backend("torch")
# config.backend("torch_gpu")
import pyscf.isdf.BackEnd.isdf_backend as BACKEND

# sys and pyscf #

import numpy as np
from pyscf import lib

from pyscf.lib.parameters import BOHR
from pyscf.pbc import df

# isdf util #

from pyscf.isdf.isdf_tools_Tsym import _kmesh_to_Kpoints, _1e_operator_gamma2k
from pyscf.isdf import isdf_tools_cell
from pyscf.isdf.isdf import ISDF
from pyscf.isdf.isdf_local import ISDF_Local

#############################

ke_cutoff = 70
basis = "gth-dzvp-molopt-sr"

boxlen = 3.57371000
prim_a = np.array([[boxlen, 0.0, 0.0], [0.0, boxlen, 0.0], [0.0, 0.0, boxlen]])
atm = [
    ["C", (0.0, 0.0, 0.0)],
    ["C", (0.8934275, 0.8934275, 0.8934275)],
    ["C", (1.786855, 1.786855, 0.0)],
    ["C", (2.6802825, 2.6802825, 0.8934275)],
    ["C", (1.786855, 0.0, 1.786855)],
    ["C", (2.6802825, 0.8934275, 2.6802825)],
    ["C", (0.0, 1.786855, 1.786855)],
    ["C", (0.8934275, 2.6802825, 2.6802825)],
]

kmeshes = [
    [1, 1, 1],
    [1, 1, 2],
    [1, 2, 2],
    [2, 2, 2],
    [2, 2, 4],
    [2, 4, 4],
    [4, 4, 4],
]  # -44.20339674 and -88.67568935
VERBOSE = 10

prim_cell = isdf_tools_cell.build_supercell(
    atm,
    prim_a,
    Ls=[1, 1, 1],
    ke_cutoff=ke_cutoff,
    basis=basis,
    pseudo="gth-pade",
    verbose=VERBOSE,
)

prim_group = [[0, 1], [2, 3], [4, 5], [6, 7]]

prim_mesh = prim_cell.mesh

for kmesh in kmeshes:

    mesh = [int(k * x) for k, x in zip(kmesh, prim_mesh)]
    print("kmesh:", kmesh, "mesh:", mesh)
    kpts = prim_cell.make_kpts(kmesh)

    direct = False
    c = 30
    rela_qr = 1e-3
    aoR_cutoff = 1e-8
    build_V_K_bunchsize = 512
    with_robust_fitting = False

    from pyscf.lib import logger
    from pyscf.lib.logger import perf_counter
    from pyscf.lib.logger import process_clock
    t0 = (process_clock(), perf_counter())

    stdout = open("out.log", "w")
    log = logger.Logger(stdout, 5)

    cell, group = isdf_tools_cell.build_supercell_with_partition(
        atm,
        prim_a,
        Ls=kmesh,
        ke_cutoff=ke_cutoff,
        partition=prim_group,
        mesh=mesh,
        basis=basis,
        pseudo="gth-pade",
        verbose=VERBOSE,
    )
    print("group:", group)

    t0 = (process_clock(), perf_counter())
    isdf = ISDF_Local(
        cell, limited_memory=True, direct=direct,
        with_robust_fitting=with_robust_fitting,
        build_V_K_bunchsize=build_V_K_bunchsize,
    )
    isdf.build(c=c, m=5, rela_cutoff=rela_qr, group=group)
    t1 = log.timer("build", *t0)

    from pyscf.pbc import scf

    mf = scf.RHF(cell)
    mf.with_df = isdf
    dm0 = mf.get_init_guess(key="minao")

    t0 = (process_clock(), perf_counter())
    vj1 = mf.get_jk(cell, dm0, with_j=True, with_k=False)[0]
    t1 = log.timer("get_j", *t0)

    t0 = (process_clock(), perf_counter())
    vk1 = mf.get_jk(cell, dm0, with_j=False, with_k=True)[1]
    t2 = log.timer("get_k", *t0)

    log.info("chk file size: %6.2e GB", 0.0)

