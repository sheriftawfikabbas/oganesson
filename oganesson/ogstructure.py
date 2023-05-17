from ase import Atoms
from ase.cell import Cell
from pymatgen.core import Structure
import numpy as np
from pymatgen.io.cif import CifParser
from oganesson.utilities import epsilon
from ase.neb import NEB
import os
from ase.io import read
from typing import Union


class OgStructure:
    '''
    The generic structure object in og.
    Unifies the type of the structure to be that of pymatgen.
    '''

    def __init__(self, structure: Union[Atoms, Structure, str] = None, file_name: str = None) -> None:
        if structure is not None:
            if isinstance(structure, str):
                parser = CifParser.from_string(structure)
                structure = parser.get_structures()
                self.structure = structure[0]
            elif isinstance(structure, Atoms):
                self.structure = self.ase_to_pymatgen(structure)
            elif isinstance(structure, Structure):
                self.structure = structure
            else:
                raise Exception('Structure type is not recognized.')
        elif file_name is not None:
            self.structure = self.ase_to_pymatgen(read(file_name))

    def __call__(self):
        return self.structure

    @staticmethod
    def db_to_structure(row):
        return Structure(coords_are_cartesian=True, coords=row.positions, species=row.symbols, lattice=row.cell)

    @staticmethod
    def distance(a, b):
        return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

    @staticmethod
    def ase_to_pymatgen(a):
        return Structure(coords_are_cartesian=True, coords=a.positions, species=a.symbols, lattice=a.cell)

    @staticmethod
    def is_image(a, b):
        return a.is_periodic_image(b)

    @staticmethod
    def pymatgen_to_ase(structure):
        fc = structure.frac_coords
        lattice = structure.lattice

        a = Atoms(scaled_positions=fc, numbers=structure.atomic_numbers, pbc=True, cell=Cell.fromcellpar(
            [lattice.a,
             lattice.b,
             lattice.c,
             lattice.alpha,
             lattice.beta,
             lattice.gamma]))

        return a

    def centerXY(self, i):
        sys = self.pymatgen_to_ase(self.structure)
        sys.positions[:, 0] = sys.positions[:, 0] - sys.positions[i, 0]
        sys.positions[:, 1] = sys.positions[:, 1] - sys.positions[i, 1]
        self.structure = self.ase_to_pymatgen(sys)
        return self

    def center(self, about_atom=None, about_point=None):
        if about_atom is not None:
            about_point = self.structure.frac_coords[about_atom]
        elif about_point is None:
            about_point = [0, 0, 0]
        sys = self.pymatgen_to_ase(self.structure)
        sys.center(about=about_point)
        self.structure = self.ase_to_pymatgen(sys)
        return self

    def equivalent_sites(self, i, site):
        if epsilon(self.structure.frac_coords[i][0] % 1, site.frac_coords[0] % 1) \
                and epsilon(self.structure.frac_coords[i][1] % 1, site.frac_coords[1] % 1) \
                and epsilon(self.structure.frac_coords[i][2] % 1, site.frac_coords[2] % 1):
            return True
        else:
            return False

    def get_site_for_neighbor_site(self, neighbor):
        for i_site in range(len(self.structure)):
            if self.equivalent_sites(i_site, neighbor):
                return i_site
        print('No equivalent:', neighbor)
        return None

    def generate_neb(self, moving_atom_species, num_images=5, r=4) -> None:
        structure = self.structure
        self.neb_paths = []
        for i_site in range(len(structure)):
            if structure[i_site].specie.symbol == moving_atom_species:
                print('Checking site', i_site)
                all_neighbors = structure.get_neighbors(
                    site=structure[i_site], r=r)
                neighbors = []
                for site in all_neighbors:
                    if site.specie.symbol == moving_atom_species:
                        neighbors += [site]
                for neighbor in neighbors:
                    i_neighbor_site = self.get_site_for_neighbor_site(
                        neighbor)
                    print(i_neighbor_site)
                    if i_neighbor_site is None:
                        raise Exception('Really? Wrong site in neighbor list!')
                    if [i_site, i_neighbor_site] in self.neb_paths or [i_neighbor_site, i_site] in self.neb_paths:
                        continue
                    else:
                        new_structure = structure.copy()
                        self.neb_paths += [[i_site, i_neighbor_site]]
                        neb_folder = 'neb_path_' + \
                            str(i_neighbor_site) + '_' + str(i_site)

                        ogs = OgStructure(new_structure)
                        new_structure = ogs.center(
                            i_site, [0.5, 0.5, 0.5]).structure

                        initial_structure = new_structure.copy()
                        final_structure = new_structure.copy()
                        initial_structure.remove_sites(
                            [i_site, i_neighbor_site])
                        initial_structure.append(
                            species=new_structure[i_site].species, coords=new_structure[i_site].frac_coords)
                        initial = OgStructure.pymatgen_to_ase(
                            initial_structure)
                        final_structure.remove_sites([i_site, i_neighbor_site])
                        final_structure.append(
                            species=new_structure[i_neighbor_site].species, coords=new_structure[i_neighbor_site].frac_coords)
                        final = OgStructure.pymatgen_to_ase(final_structure)

                        self.images = [initial]
                        self.images += [initial.copy()
                                        for i in range(num_images)]
                        self.images += [final]
                        self.neb = NEB(self.images)
                        os.mkdir(neb_folder)
                        # Interpolate linearly the potisions of the three middle images:
                        self.neb.interpolate(mic=True)
                        for i in range(len(self.images)):
                            image_str = OgStructure.ase_to_pymatgen(
                                self.images[i]).to(fmt='poscar')
                            f = open(neb_folder+'/'+str(i).zfill(2), 'w')
                            f.write(image_str)
                            f.close()
