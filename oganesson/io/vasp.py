from oganesson.ogstructure import OgStructure
from pymatgen.core import Structure
import numpy as np

def outcar_extractor(outcar_directory:str=".",outcar_file:str='OUTCAR'):
    def get_indices(l, s):
        indices = []
        for i in range(len(l)):
            if s in l[i]:
                indices += [i]
        return indices

    def get_vectors(l, positions, displacement, rows, columns):
        vectors = []
        for position in positions:
            vector = []
            s = l[position+displacement:position+displacement+rows]
            for r in s:
                c = r.split()
                c = c[0:columns]
                c = [float(x) for x in c]
                vector += [c]
            vectors += [vector]
        return vectors

    def get_atoms_counts(l):
        ion_counts = []
        ions = []
        for s in l:
            if 'ions per type' in s:
                s = s.replace('ions per type', '').replace('=', '').split()
                ion_counts = [int(x) for x in s]
                break
        for s in l:
            if 'POSCAR =' in s:
                s = s.replace('POSCAR =', '').split()
                ions = s
        return ion_counts, ions

    outcarf = open(outcar_directory+outcar_file, 'r')
    outcar = outcarf.readlines()
    outcarf.close()

    structures = []
    ion_counts, ions = get_atoms_counts(outcar)
    atoms = []
    for i in range(len(ions)):
        atoms += ion_counts[i]*[ions[i]]

    number_of_atoms = sum(ion_counts)

    direct_lattice_vectors_positions = get_indices(
        outcar, 'direct lattice vectors')
    positions_forces_vectors_positions = get_indices(
        outcar, 'TOTAL-FORCE (eV/Angst)')
    stresses_positions = get_indices(
        outcar, 'stress matrix')
    lattice_vectors = get_vectors(
        outcar, direct_lattice_vectors_positions, 1, 3, 3)
    positions_forces_vectors = np.array(get_vectors(
        outcar, positions_forces_vectors_positions, 2, number_of_atoms, 6))
    positions_vectors = positions_forces_vectors[:, :, 0:3]
    forces_vectors = positions_forces_vectors[:, :, 3:6]
    stress_matrices = get_vectors(
        outcar, stresses_positions, 1, 3, 3)
    for i in range(len(positions_vectors)):
        structures += [OgStructure(Structure(species=atoms,
                                             coords=positions_vectors[i], lattice=lattice_vectors[i]))]

    return structures, forces_vectors, stress_matrices
