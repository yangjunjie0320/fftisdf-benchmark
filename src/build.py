import os, sys, numpy, scipy
API_KEY = os.getenv("MP_API_KEY", None)
TMPDIR = os.getenv("TMPDIR", "/tmp")
assert os.path.exists(TMPDIR), f"TMPDIR {TMPDIR} does not exist"

def download_poscar(mid: str, name=None, path=None, is_conventional=False,
                    supercell_factor=None):
    from mp_api.client import MPRester as M
    if path is None:
        path = TMPDIR

    if name is None:
        name = f"{mid}"

    name = name + "-conv" if is_conventional else name + "-prim"
    # Initialize the Materials Project REST API client
    with M(API_KEY) as mpr:
        structure = mpr.get_structure_by_material_id(mid, conventional_unit_cell=is_conventional)
        tmp = os.path.join(TMPDIR, f"{name}.vasp")
        structure.to(filename=str(tmp), fmt="poscar")
        print(f"\nSuccessfully downloaded POSCAR for {mid} to {tmp}")
        
        import ase
        atoms = ase.io.read(tmp)
        poscar_path = os.path.join(path, f"{name}.vasp")

        if supercell_factor:
            atoms = atoms * supercell_factor
            poscar_path = os.path.join(path, f"{name}-{'x'.join(map(str, supercell_factor))}.vasp")
        
        atoms.write(poscar_path, format="vasp")
        return str(poscar_path)
    
def ase_atoms_to_pyscf(ase_atoms):                                                                      
    return [[atom.symbol, atom.position] for atom in ase_atoms]
    
def cell_from_poscar(poscar_file: str):
    import ase
    from ase.io import read
    atoms = read(poscar_file)

    from pyscf.pbc import gto
    c = gto.Cell()
    c.atom = ase_atoms_to_pyscf(atoms)
    c.a = numpy.array(atoms.cell)
    c.exp_to_discard = 1e-10
    c.unit = 'A'
    return c

# Example usage:
if __name__ == "__main__":
    path = "/Users/yangjunjie/work/isdf-benchmark/data/"
    data = [("mp-66", "diamond"), ("mp-19009", "nio"), ("mp-390", "tio2"), ("mp-4826", "cacuo2")]

    for m, n in data:
        for is_conventional in [True, False]:
            poscar_file = download_poscar(
                m, name=n, path=path, is_conventional=is_conventional,
                supercell_factor=None
            )
            cell = cell_from_poscar(poscar_file)
            cell.basis = 'gth-szv-molopt-sr'
            cell.pseudo = 'gth-pade'
            cell.build()
            print(cell)

            poscar_file = download_poscar(
                m, name=n, path=path, is_conventional=is_conventional, 
                supercell_factor=(2, 2, 1)
            )
            cell = cell_from_poscar(poscar_file)
            cell.basis = 'gth-szv-molopt-sr'
            cell.pseudo = 'gth-pade'
            cell.build()
            print(cell)
            