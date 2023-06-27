from pymatgen.core import Structure
import numpy as np

class Outcar:
    def __init__(self, outcar_directory:str="./", outcar_file:str='OUTCAR', poscar_file:str=None) -> None:
        self.outcar_directory = outcar_directory
        self.outcar_file = outcar_file
        self.poscar_file = poscar_file

    def get_md_data(self):
        return self._outcar_extractor()
    
    def write_md_data(self, file_name:str = None, path:str=None):
        structures, forces_vectors, stress_matrices = self._outcar_extractor()
        if file_name is None:
            file_name = self.outcar_file+'.json'
        if path is None:
            path = self.outcar_directory
        output = {'structures':structures,'forces':forces_vectors, 'stresses':stress_matrices}
        import json
        with open(path+file_name,'w') as f:
            json.dump(output, f)

    def _outcar_extractor(self):
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

        outcarf = open(self.outcar_directory+self.outcar_file, 'r')
        outcar = outcarf.readlines()
        outcarf.close()

        structures = []
        atoms = []
        if self.poscar_file is not None:
            structure = Structure.from_file(self.outcar_directory+self.poscar_file)
            number_of_atoms = len(structure)
            atoms = [s.specie.symbol for s in structure]
        else:
            ion_counts, ions = get_atoms_counts(outcar)
        
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
            structures += [Structure(species=atoms,
                                                coords=positions_vectors[i], lattice=lattice_vectors[i], coords_are_cartesian=True).as_dict()]

        return structures, forces_vectors.tolist(), stress_matrices
