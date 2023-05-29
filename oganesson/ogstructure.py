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
from oganesson.utilities.bonds_dictionary import bonds_dictionary
from m3gnet.models import Relaxer


class OgStructure:
    '''
    The generic structure object in og.
    Unifies the type of the structure to be that of pymatgen.
    '''

    def __init__(self, structure: Union[Atoms, Structure, str] = None, file_name: str = None) -> None:
        if structure is not None:
            if isinstance(structure, OgStructure):
                self = structure
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

    def to_ase(self):
        fc = self.structure.frac_coords
        lattice = self.structure.lattice

        a = Atoms(scaled_positions=fc, numbers=self.structure.atomic_numbers, pbc=True, cell=Cell.fromcellpar(
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

    def _get_site_for_neighbor_site(self, neighbor):
        for i_site in range(len(self.structure)):
            if self.equivalent_sites(i_site, neighbor):
                return i_site
        print('No equivalent:', neighbor)
        return None

    def _get_min_max_bonds(self, atom_1_number, atom_2_number):
        key = str(atom_1_number) + '_' + \
            str(atom_2_number) if atom_1_number <= atom_2_number else str(
                atom_2_number) + '_' + str(atom_1_number)
        return bonds_dictionary['min'][key], bonds_dictionary['max'][key]

    def _nudgeVector(self, atom_coords, n, min_bond, max_bond):
        atom_coords = np.array(atom_coords)
        n = np.array(n)
        if self.distance(atom_coords, n) < min_bond:
            return (atom_coords-n)/np.linalg.norm(atom_coords-n)/100
        elif self.distance(atom_coords, n) > max_bond:
            return -(atom_coords-n)/np.linalg.norm(atom_coords-n)/100

    @staticmethod
    def is_image(a, b):
        return a.is_periodic_image(b)

    def add_atom_to_surface(self, atom_symbol, max_trials=1000):
        s = self.to_ase()
        path = './'
        s.positions[:, 2] = s.positions[:, 2] - s.positions[:, 2].min()
        nudge = np.array([0.0, 0.0, 0.0])

        ii = 0
        history = ""
        do_leap = False
        while ii < max_trials:
            s1 = s
            s1_thickness = s1.positions[:, 2].max() - s1.positions[:, 2].min()
            if do_leap:
                atom = Atoms(positions=[[s.cell.cellpar()[0]*np.random.random(), s.cell.cellpar()[1]*np.random.random(), nudge[2]+s1_thickness+2.5/2]],
                             pbc=True, cell=s.cell, symbols=[atom_symbol])

            else:
                atom = Atoms(positions=[[s.cell.cellpar()[0]/2 + nudge[0], s.cell.cellpar()[1]/2 + nudge[1], nudge[2]+s1_thickness+2.5/2]],
                             symbols=[atom_symbol], pbc=True, cell=s.cell)

            intercalation = s1 + atom

            intercalation = self.ase_to_pymatgen(intercalation)

            n = intercalation.get_neighbors(intercalation[-1], 3.5)
            n_Li_distances = []
            needs_nudging = False
            if len(n) == 0:
                nudge[2] += -0.01
            else:
                for n_Li in n:
                    n_Li_distances += [self.distance(intercalation[-1].coords,
                                                     n_Li.coords)]
                    history += str([n_Li_distances])+'\n'
                    min_bond, max_bond = self._get_min_max_bonds(
                        intercalation[-1].specie.Z, n_Li.specie.Z)
                    if self.distance(intercalation[-1].coords, n_Li.coords) < min_bond or self.distance(intercalation[-1].coords, n_Li.coords) > max_bond and not self.is_image(n_Li, intercalation[-1]):
                        nudge += self._nudgeVector(
                            intercalation[-1].coords, n_Li.coords, min_bond, max_bond)
                        needs_nudging = True
                if not needs_nudging:
                    return OgStructure(intercalation)
                elif ii % 1000 == 0:
                    do_leap = True
                elif ii == max_trials-1:
                    intercalation.to('cif', path+'bad_'+str(id)+'.cif')
                    print('bad', id)
                    f = open(path+'bad_'+str(id)+'.history', 'w')
                    f.write(history)
                    f.close()
                    return False
            ii += 1
        return False

    def relax(self, relaxation_method='m3gnet', cellbounds=None):
        relaxer = Relaxer()
        relax_results = relaxer.relax(self.structure, verbose=True)
        self.structure = relax_results['final_structure']

    def generate_neb(self, moving_atom_species, num_images=5, r=3, relaxation_method=None) -> None:
        structure = self.structure
        self.neb_paths = []
        for i_site in range(len(structure)):
            if structure[i_site].specie.symbol == moving_atom_species:
                all_neighbors = structure.get_neighbors(
                    site=structure[i_site], r=r)
                neighbors = []
                for site in all_neighbors:
                    if site.specie.symbol == moving_atom_species:
                        neighbors += [site]
                print('Checking site', i_site,
                      ': Surrounded by', len(neighbors))
                for neighbor in neighbors:
                    i_neighbor_site = self._get_site_for_neighbor_site(
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
                        self.neb.interpolate(mic=True)
                        for i in range(len(self.images)):
                            self.images[i] = OgStructure(self.images[i])
                            if relaxation_method == 'm3gnet':
                                self.images[i] = self.images[i].relax()
                            image_str = self.images[i].structure.to(
                                fmt='poscar')
                            f = open(neb_folder+'/'+str(i).zfill(2), 'w')
                            f.write(image_str)
                            f.close()

   
