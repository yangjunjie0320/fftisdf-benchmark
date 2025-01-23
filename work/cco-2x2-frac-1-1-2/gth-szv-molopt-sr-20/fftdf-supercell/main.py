import os, sys
import numpy, scipy
import pyscf

TMPDIR = os.getenv("TMPDIR", None)
DATA_PATH = os.getenv("DATA_PATH", None)
PYSCF_MAX_MEMORY = os.getenv("PYSCF_MAX_MEMORY", 4000)
PYSCF_MAX_MEMORY = int(PYSCF_MAX_MEMORY)

def main(args):
    from utils import cell_from_poscar
    path = os.path.join(DATA_PATH, args.cell)
    assert os.path.exists(path), "Cell file not found: %s" % path

    c = cell_from_poscar(path)
    c.basis = args.basis
    c.pseudo = args.pseudo
    c.verbose = 0
    c.unit = 'aa'
    # c.exp_to_discard = 0.1
    # c.exp_to_discard = 0.0
    c.max_memory = PYSCF_MAX_MEMORY
    c.ke_cutoff = args.ke_cutoff
    c.build(dump_input=False)

    kmesh = [int(x) for x in args.kmesh.split("-")]
    kmesh = numpy.array(kmesh)
    kpts = c.get_kpts(kmesh)

    pcell = c.copy(deep=True)

    natm = len(pcell.atom)
    atm = []
    for ia in range(natm):
        s = pcell.atom_symbol(ia)
        x = pcell.atom_coord(ia, unit='A')
        atm.append([s, x])

    from pyscf.lib.parameters import BOHR
    lvx = pcell.lattice_vectors() * BOHR

    from pyscf.isdf.isdf_tools_cell import build_supercell_with_partition
    scell, group = build_supercell_with_partition(
        atm, lvx, spin=0, charge=0, mesh=None, Ls=kmesh,
        partition=None, basis=args.basis,
        pseudo=args.pseudo, ke_cutoff=args.ke_cutoff,
        max_memory=PYSCF_MAX_MEMORY, 
        precision=1e-12, 
        use_particle_mesh_ewald=True,
        verbose=4,
    )

    from pyscf.pbc.df.fft import FFTDF
    df_obj = FFTDF(scell)
    df_obj.verbose = 10

    from utils import get_jk_time
    kmesh = None
    tmp = os.path.join(TMPDIR, "fftdf.chk")
    os.system("touch %s" % tmp)
    
    get_jk_time(scell, kmesh=kmesh, df_obj=df_obj, tmp=tmp)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=str, default="diamond-prim.vasp")
    parser.add_argument("--kmesh", type=str, default="2-2-2")
    parser.add_argument("--ke_cutoff", type=float, default=200)
    parser.add_argument("--basis", type=str, default="gth-dzvp-molopt-sr")
    parser.add_argument("--pseudo", type=str, default="gth-pade")

    args = parser.parse_args()
    main(args)
