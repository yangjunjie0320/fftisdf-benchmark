# backend to test #

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
basis = "gth-dzvp"

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
    [1, 1, 3],
    [1, 1, 4],
    [1, 2, 2],
    [2, 2, 2],
    [2, 2, 3],
    [2, 3, 3],
    [3, 3, 3],
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
    cell.max_memory = 200
    print("group:", group)

    isdf = ISDF_Local(
        cell, with_robust_fitting=True, limited_memory=True, build_V_K_bunchsize=56
    )
    isdf.build(c=30, m=5, rela_cutoff=1e-4, group=group)
    # isdf.force_translation_symmetry(kmesh)

    from pyscf.pbc import scf

    mf = scf.RHF(cell)
    mf.with_df = isdf
    mf.kernel()
