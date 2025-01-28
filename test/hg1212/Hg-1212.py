#!/usr/bin/env python

"""
Hg-1212 using 2x2 cell.
"""

from mpi4pyscf.tools import mpi
import os, sys
import time
import numpy as np
import scipy.linalg as la
from pyscf.pbc import gto, scf, df, dft, cc
from pyscf import lib
from pyscf.pbc.lib import chkfile
from libdmet.system import lattice
from libdmet.utils.misc import mdot, max_abs, read_poscar
import libdmet.utils.logger as log

log.verbose = "DEBUG1"

start = time.time()

# this is the number of doped holes
nelec_dop = 0

cell_0 = read_poscar(fname="./Hg1212-2x2.vasp")

cell_0.basis = 'cc-pvdz.dat'
cell_0.pseudo = {
               'Cu1': 'GTH-PBE-q19', 
               'Cu2': 'GTH-PBE-q19',
               'O1': 'gth-pbe',
               'O2': 'gth-pbe',
               'Cu3': 'GTH-PBE-q19', 
               'Cu4': 'GTH-PBE-q19',
               'O3': 'gth-pbe',
               'O4': 'gth-pbe',
               'Hg': 'gth-pbe', 
               'Ba': 'gth-pbe',
               'Ca': 'gth-pbe'}

kmesh = [4, 4, 1]
cell_0.spin = 0
cell_0.verbose = 5
cell_0.max_memory = 160000
cell_0.precision = 1e-15
cell_0.build()

Lat_0 = lattice.Lattice(cell_0, kmesh)

nelec0 = int(cell_0.nelectron)


from libdmet.utils import match_lattice_orbitals
from pyscf.pbc import tools
cell_pm = read_poscar("Hg1212-PM.vasp")

cell_pm.basis = 'cc-pvdz.dat'
cell_pm.pseudo = {
               'Cu1': 'GTH-PBE-q19', 
               'Cu2': 'GTH-PBE-q19',
               'O1': 'gth-pbe',
               'O2': 'gth-pbe',
               'Cu3': 'GTH-PBE-q19', 
               'Cu4': 'GTH-PBE-q19',
               'O3': 'gth-pbe',
               'O4': 'gth-pbe',
               'Hg': 'gth-pbe', 
               'Ba': 'gth-pbe',
               'Ca': 'gth-pbe'}

cell_pm.spin = 0
cell_pm.verbose = 5
cell_pm.max_memory = 160000
cell_pm.precision = 1e-15
cell_pm.build()

kmesh = [8, 8, 1]
Lat_pm = lattice.Lattice(cell_pm, kmesh)

idx = match_lattice_orbitals(Lat_pm.bigcell, Lat_0.bigcell)

nkpts = Lat_0.nkpts
doping = nelec_dop / (nkpts * 8)
rdm1_pm = np.load("../../doping_PBE_rba/x_%.2f/rdm1.npy"%doping)
rdm1_pm = Lat_pm.expand(Lat_pm.k2R(rdm1_pm))

rdm1_pm = rdm1_pm[np.ix_(idx, idx)]
rdm1 = Lat_0.extract_stripe(rdm1_pm)
rdm1_pm = None
rdm1 = Lat_0.R2k(rdm1)
dm0 = rdm1


hcore_pm = np.load("../../doping_PBE_rba/x_%.2f/hcore.npy"%doping)
hcore_pm = Lat_pm.expand(Lat_pm.k2R(hcore_pm))

hcore_pm = hcore_pm[np.ix_(idx, idx)]
hcore_new = Lat_0.extract_stripe(hcore_pm)
hcore_pm = None
hcore_new = Lat_0.R2k(hcore_new)


cell_0 = None
Lat_0 = None
cell_pm = Lat_pm = None


cell = read_poscar(fname="./Hg1212-2x2.vasp")

cell.basis = 'cc-pvdz.dat'
cell.pseudo = {
               'Cu1': 'GTH-PBE-q19', 
               'Cu2': 'GTH-PBE-q19',
               'O1': 'gth-pbe',
               'O2': 'gth-pbe',
               'Cu3': 'GTH-PBE-q19', 
               'Cu4': 'GTH-PBE-q19',
               'O3': 'gth-pbe',
               'O4': 'gth-pbe',
               'Hg': 'gth-pbe', 
               'Ba': 'gth-pbe',
               'Ca': 'gth-pbe'}

kmesh = [4, 4, 1]
cell.spin = 0
cell.verbose = 5
cell.max_memory = 160000
cell.precision = 1e-15
cell.build()

nelec0 = int(cell.nelectron)

Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts
nao = Lat.nao
nkpts = Lat.nkpts

doping = nelec_dop / (nkpts * 8)

charges_old = np.array(cell.atom_charges(), dtype=float)
z_O = charges_old[-1]
n_O7 = 4
occ = -(nelec_dop / (n_O7 * z_O * nkpts))
#cell.nelectron = nelec0 + (occ * n_O7 * z_O)
cell.nelectron = nelec0 - (nelec_dop / nkpts)

Cu_3d_A = np.append(cell.search_ao_label("Cu1 3dx2-y2"), cell.search_ao_label("Cu4 3dx2-y2"))
Cu_3d_B = np.append(cell.search_ao_label("Cu2 3dx2-y2"), cell.search_ao_label("Cu3 3dx2-y2"))
##O7_idx = cell.search_ao_label("O7.*")
#O_3band = np.hstack((cell.search_ao_label("4 O1 2py"), cell.search_ao_label("5 O1 2px"), \
#         cell.search_ao_label("6 O1 2py")  , cell.search_ao_label("7 O1 2px"), \
#         cell.search_ao_label("8 O1 2px")  , cell.search_ao_label("9 O1 2py"), \
#         cell.search_ao_label("10 O1 2px") , cell.search_ao_label("11 O1 2py")))

#kpts_symm_0 = cell_0.make_kpts(kmesh, with_gamma_point=True, wrap_around=True, 
#                           space_group_symmetry=True, time_reversal_symmetry=True)
kpts_symm = cell.make_kpts(kmesh, with_gamma_point=True, wrap_around=True, 
                           space_group_symmetry=True, time_reversal_symmetry=True)

gdf_fname = '../../../PBE0_ox/gdf_ints_Hg1212_441.h5'
gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = gdf_fname
gdf.use_mpi = True

kmf = dft.KUKS(cell, kpts_symm).density_fit()

kmf.xc = 'pbe0'
kmf.grids.level = 5
kmf.exxdiv = None

#from pyscf.scf import addons
#kmf = addons.remove_linear_dep_(kmf, threshold=1e-6, lindep=1e-6)

from libdmet.routine import pbc_helper as pbc_hp
from pyscf.data.nist import HARTREE2EV
sigma = 0.2 / HARTREE2EV
kmf = pbc_hp.smearing_(kmf, sigma=sigma, method="fermi", tol=1e-12)

#dm0 = kmf.get_init_guess()

#kmf = scf.addons.smearing_(kmf, sigma=0.005, method="fermi")
kmf.with_df = gdf
kmf.with_df._cderi = gdf_fname
kmf.conv_tol = 1e-9
chk_fname = './Hg1212_UPBE0.chk'
kmf.chkfile = chk_fname
kmf.diis_space = 15
kmf.max_cycle = 150
kmf.diis_start_cycle = 0

#dm0 = np.asarray(kmf.get_init_guess(key='atom'))
#
#for dm in dm0:
#    dm[np.ix_(O7_idx, O7_idx)] = 0.0

#dm0 = np.load("rdm1_symm.npy") * 0.5
#dm0 = np.asarray((dm0, dm0))
#dm0[0, :, Cu_3d_A, Cu_3d_A] *= 2.0
#dm0[0, :, Cu_3d_B, Cu_3d_B]  = 0.0
#dm0[1, :, Cu_3d_A, Cu_3d_A]  = 0.0
#dm0[1, :, Cu_3d_B, Cu_3d_B] *= 2.0
#dm0 = np.load("../rdm1_UPBE0.npy")

#dm0 = np.load("rdm1.npy")
#dm0 = dm0[kpts_symm.ibz2bz]

#kmf_res = dft.KRKS(cell_0, kpts_symm_0).density_fit()
#data = chkfile.load("../../doping_PBE_vca/x_%.2f_2x2/Hg1212_RPBE.chk"%doping, "scf")

#hcore_new = np.load("../../../PBE0/doping_PBE_vca/x_%.2f/hcore_new.npy"%doping)
#hcore_new = kpts_symm_0.transform_dm(hcore_new)
hcore_new = hcore_new[kpts_symm.ibz2bz]

kmf.get_hcore = lambda *args: hcore_new

#kmf_res.__dict__.update(data)
#dm0 = np.asarray(kmf_res.make_rdm1())
#dm0 = kpts_symm_0.transform_dm(dm0) * 0.5
#dm0 = np.asarray((dm0, dm0))
dm0 = np.asarray((dm0, dm0)) * 0.5

from libdmet.basis_transform import make_basis
from libdmet.lo import lowdin
kmf_no_symm = pbc_hp.kmf_symm_(kmf)
C_ao_lo = lowdin.lowdin_k(kmf_no_symm, pre_orth_ao='SCF')

dm0_lo = make_basis.transform_rdm1_to_lo(dm0, C_ao_lo, kmf_no_symm.get_ovlp())
dm0_lo_R = Lat.k2R(dm0_lo)
Lat.mulliken_lo_R0(dm0_lo_R[:, 0])

dm0_lo_R[0, 0, Cu_3d_A, Cu_3d_A]  = np.minimum(dm0_lo_R[0, 0, Cu_3d_A, Cu_3d_A] * 2.0, 1.0)
dm0_lo_R[0, 0, Cu_3d_B, Cu_3d_B]  = 0.0
dm0_lo_R[1, 0, Cu_3d_A, Cu_3d_A]  = 0.0
dm0_lo_R[1, 0, Cu_3d_B, Cu_3d_B]  = np.minimum(dm0_lo_R[1, 0, Cu_3d_B, Cu_3d_B] * 2.0, 1.0)

dm0_lo = Lat.R2k(dm0_lo_R)
dm0 = make_basis.transform_rdm1_to_ao(dm0_lo, C_ao_lo)

dm0_lo = make_basis.transform_rdm1_to_lo(dm0, C_ao_lo, kmf_no_symm.get_ovlp())
dm0_lo_R = Lat.k2R(dm0_lo)

print ("after polarization")
Lat.mulliken_lo_R0(dm0_lo_R[:, 0])

dm0 = dm0[:, kpts_symm.ibz2bz]



#dm0 = np.asarray(kmf.get_init_guess(key='atom'))
#
#for dm in dm0:
#    dm[np.ix_(O7_idx, O7_idx)] = 0.0

#dm0 = np.load("rdm1_symm.npy") * 0.5
#dm0 = np.asarray((dm0, dm0))
#dm0[0, :, Cu_3d_A, Cu_3d_A] *= 2.0
#dm0[0, :, Cu_3d_B, Cu_3d_B]  = 0.0
#dm0[1, :, Cu_3d_A, Cu_3d_A]  = 0.0
#dm0[1, :, Cu_3d_B, Cu_3d_B] *= 2.0
#dm0 = np.load("../rdm1_UPBE0.npy")

#dm0 = dm0[kpts_symm.ibz2bz]


# VCA

#atom_idx = np.arange(len(charges_old) - n_O7, len(charges_old))
#vnuc_vca = pbc_hp.get_veff_vca(gdf, atom_idx, occ, kpts_symm=kpts_symm)
#hcore_full = kmf.get_hcore()
#np.save("hcore_full.npy", hcore_full)
#hcore_new = hcore_full + vnuc_vca
#kmf.get_hcore = lambda *args: hcore_new

#hcore_new = np.load("hcore_symm.npy")
#kmf.get_hcore = lambda *args: hcore_new

kmf.kernel(dm0=dm0)

ovlp = kmf.get_ovlp()
np.save("ovlp_symm.npy", ovlp)
ovlp = kpts_symm.transform_fock(ovlp)
np.save("ovlp.npy", ovlp)

rdm1 = kmf.make_rdm1()
np.save("rdm1_symm.npy", rdm1)
rdm1 = kpts_symm.transform_dm(rdm1)
np.save("rdm1.npy", rdm1)

hcore = kmf.get_hcore()
np.save("hcore_symm.npy", hcore)
hcore = kpts_symm.transform_fock(hcore)
np.save("hcore.npy", hcore)

#fock = kmf.get_fock()
#np.save("fock_symm.npy", fock)
#fock = kpts_symm.transform_fock(fock)
#np.save("fock.npy", fock)


kmf = pbc_hp.kmf_symm_(kmf)

Lat.analyze(kmf, pre_orth_ao='SCF')
