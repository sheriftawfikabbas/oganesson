import numpy as np
from ase.neb import NEB
import os
import math
from ase.io import read
from typing import List, Optional, Tuple, Union
from numpy.typing import ArrayLike
import random
import uuid

import matplotlib.pyplot as plt
from ase import Atoms, Atom
from ase.cell import Cell
from pymatgen.core import Structure, Element
from bsym.interface.pymatgen import unique_structure_substitutions
from m3gnet.models import Relaxer
from pymatgen.io.cif import CifParser
from pymatgen.core import Lattice

from diffusivity.diffusion_coefficients import DiffusionCoefficient
from ase.md.md import Trajectory
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.util.coord import pbc_shortest_vectors
from m3gnet.graph._compute import *
from m3gnet.graph._converters import RadiusCutoffGraphConverter
from sympy import integrate, cos, pi, sqrt, symbols, solve, sin
from oganesson.molecular_dynamics import MolecularDynamics
from oganesson.utilities import epsilon
from oganesson.utilities.constants import F
from oganesson.utilities.bonds_dictionary import bonds_dictionary
from oganesson.utilities import atomic_data


class OgStructure:
    """
    The generic structure object in og.
    Unifies the type of the structure to be that of pymatgen.
    """

    def __init__(
        self,
        structure: Union[Atoms, Structure, str] = None,
        file_name: str = None,
        structure_tag=None,
    ) -> None:
        self.structure_tag = structure_tag
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
                raise Exception("Structure type is not recognized.")
        elif file_name is not None:
            self.structure = self.ase_to_pymatgen(read(file_name))

        self.structure.sort()

    def __call__(self):
        return self.structure

    def __len__(self) -> int:
        """
        Returns number of atoms in structure.
        """
        return len(self.structure)

    @staticmethod
    def db_to_structure(row):
        return Structure(
            coords_are_cartesian=True,
            coords=row.positions,
            species=row.symbols,
            lattice=row.cell,
        )

    @staticmethod
    def distance(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    @staticmethod
    def ase_to_pymatgen(a):
        return Structure(
            coords_are_cartesian=True,
            coords=a.positions,
            species=a.symbols,
            lattice=a.cell,
        )

    @staticmethod
    def is_image(a, b):
        return a.is_periodic_image(b)

    @staticmethod
    def pymatgen_to_ase(structure):
        fc = structure.frac_coords
        lattice = structure.lattice

        a = Atoms(
            scaled_positions=fc,
            numbers=structure.atomic_numbers,
            pbc=True,
            cell=Cell.fromcellpar(
                [
                    lattice.a,
                    lattice.b,
                    lattice.c,
                    lattice.alpha,
                    lattice.beta,
                    lattice.gamma,
                ]
            ),
        )

        return a

    def is_transition_metal(self):
        for a in self.structure.atomic_numbers:
            if a not in atomic_data.d_groups_flat:
                return False
        return True

    def to_ase(self):
        fc = self.structure.frac_coords
        lattice = self.structure.lattice

        a = Atoms(
            scaled_positions=fc,
            numbers=self.structure.atomic_numbers,
            pbc=True,
            cell=Cell.fromcellpar(
                [
                    lattice.a,
                    lattice.b,
                    lattice.c,
                    lattice.alpha,
                    lattice.beta,
                    lattice.gamma,
                ]
            ),
        )

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

    def sort_species(self):
        return OgStructure(self.structure.get_sorted_structure())

    def get_bonds(self, bond_data=None):
        if bond_data is None:
            bond_data = {}
        for c1 in self.structure:
            c = self.structure.get_neighbors(site=c1, r=4)
            for c2 in c:
                Rij = self.distance(c1.coords, c2.coords)

                if c1.specie.number < c2.specie.number:
                    k = str(c1.specie.number) + "_" + str(c2.specie.number)
                else:
                    k = str(c2.specie.number) + "_" + str(c1.specie.number)

                if k in bond_data.keys():
                    available = False
                    for bb in bond_data[k]:
                        if abs(bb - Rij) < 1e-5:
                            available = True
                            break
                    if not available:
                        bond_data[k] += [Rij]
                else:
                    bond_data[k] = [Rij]
        return bond_data

    def get_bonds_blocks(self, bond_data=None):

        if bond_data is None:
            bond_data = {}
        for c1 in self.structure:
            c = self.structure.get_neighbors(site=c1, r=4)
            for c2 in c:
                Rij = self.distance(c1.coords, c2.coords)

                g1 = atomic_data.get_group(c1.specie.number)
                g2 = atomic_data.get_group(c2.specie.number)

                if g1 < g2:
                    k = str(g1) + "_" + str(g2)
                else:
                    k = str(g2) + "_" + str(g1)

                if k in bond_data.keys():
                    available = False
                    for bb in bond_data[k]:
                        if abs(bb - Rij) < 1e-5:
                            available = True
                            break
                    if not available:
                        bond_data[k] += [Rij]
                else:
                    bond_data[k] = [Rij]
        return bond_data

    def translate(self, v):
        self.structure.translate_sites(range(len(self)), v)

    def equivalent_sites(self, i, site):
        if (
            epsilon(self.structure.frac_coords[i][0] % 1, site.frac_coords[0] % 1)
            and epsilon(self.structure.frac_coords[i][1] % 1, site.frac_coords[1] % 1)
            and epsilon(self.structure.frac_coords[i][2] % 1, site.frac_coords[2] % 1)
        ):
            return True
        else:
            return False

    def _get_site_for_neighbor_site(self, neighbor):
        for i_site in range(len(self.structure)):
            if self.equivalent_sites(i_site, neighbor):
                return i_site
        print("og:No equivalent:", neighbor)
        return None

    def _get_min_max_bonds(self, atom_1_number, atom_2_number):
        key = (
            str(atom_1_number) + "_" + str(atom_2_number)
            if atom_1_number <= atom_2_number
            else str(atom_2_number) + "_" + str(atom_1_number)
        )
        return bonds_dictionary["min"][key], bonds_dictionary["max"][key]

    def _nudgeVector(self, atom_coords, n, min_bond, max_bond):
        atom_coords = np.array(atom_coords)
        n = np.array(n)
        if self.distance(atom_coords, n) < min_bond:
            return (atom_coords - n) / np.linalg.norm(atom_coords - n) / 100
        elif self.distance(atom_coords, n) > max_bond:
            return -(atom_coords - n) / np.linalg.norm(atom_coords - n) / 100

    def add_atom_to_surface(self, atom_symbol, max_trials=1000):
        s = self.to_ase()
        s.positions[:, 2] = s.positions[:, 2] - s.positions[:, 2].min()
        nudge = np.array([0.0, 0.0, 0.0])

        ii = 0
        history = ""
        do_leap = False
        while ii < max_trials:
            s1 = s
            s1_thickness = s1.positions[:, 2].max() - s1.positions[:, 2].min()
            if do_leap:
                atom = Atoms(
                    positions=[
                        [
                            s.cell.cellpar()[0] * np.random.random(),
                            s.cell.cellpar()[1] * np.random.random(),
                            nudge[2] + s1_thickness + 2.5 / 2,
                        ]
                    ],
                    pbc=True,
                    cell=s.cell,
                    symbols=[atom_symbol],
                )

            else:
                atom = Atoms(
                    positions=[
                        [
                            s.cell.cellpar()[0] / 2 + nudge[0],
                            s.cell.cellpar()[1] / 2 + nudge[1],
                            nudge[2] + s1_thickness + 2.5 / 2,
                        ]
                    ],
                    symbols=[atom_symbol],
                    pbc=True,
                    cell=s.cell,
                )

            intercalation = s1 + atom

            intercalation = self.ase_to_pymatgen(intercalation)

            n = intercalation.get_neighbors(intercalation[-1], 3.5)
            n_Li_distances = []
            needs_nudging = False
            if len(n) == 0:
                nudge[2] += -0.01
            else:
                for n_Li in n:
                    n_Li_distances += [
                        self.distance(intercalation[-1].coords, n_Li.coords)
                    ]
                    history += str([n_Li_distances]) + "\n"
                    min_bond, max_bond = self._get_min_max_bonds(
                        intercalation[-1].specie.Z, n_Li.specie.Z
                    )
                    if (
                        self.distance(intercalation[-1].coords, n_Li.coords) < min_bond
                        or self.distance(intercalation[-1].coords, n_Li.coords)
                        > max_bond
                        and not self.is_image(n_Li, intercalation[-1])
                    ):
                        nudge += self._nudgeVector(
                            intercalation[-1].coords, n_Li.coords, min_bond, max_bond
                        )
                        needs_nudging = True
                if not needs_nudging:
                    return OgStructure(intercalation)
                elif ii % 1000 == 0:
                    do_leap = True
                elif ii == max_trials - 1:
                    return False
            ii += 1
        return False

    def adsorption_scanner(self, atom_symbol, max_trials=1000):
        def available_in_list(p, l):
            for il in l:
                if self.distance(il, p) < 0.5:
                    return True
            return False

        s = self.to_ase()
        s_exhibit = self.to_ase()
        partition = 0.5
        number_of_partitions_a = int(s.cell.cellpar()[0] / partition)
        number_of_partitions_b = int(s.cell.cellpar()[1] / partition)
        adsorption_positions = []
        adsorption_structures = []
        for a in np.linspace(
            0, s.cell.cellpar()[0] - partition, number_of_partitions_a
        ):
            for b in np.linspace(
                0, s.cell.cellpar()[1] - partition, number_of_partitions_b
            ):
                structure = self._adsorption_scanner_position(
                    atom_symbol, [a, b], max_trials
                )
                if structure:
                    new_position = structure().cart_coords[-1]
                    if not available_in_list(new_position, adsorption_positions):
                        adsorption_positions += [new_position]
                        adsorption_structures += [structure]
                        s_exhibit.append(
                            Atom(symbol=atom_symbol, position=new_position)
                        )
        adsorption_structures += [OgStructure(s_exhibit)]
        return adsorption_structures

    def _adsorption_scanner_position(self, atom_symbol, position, max_trials=1000):
        s = self.to_ase()
        s.positions[:, 2] = s.positions[:, 2] - s.positions[:, 2].min()
        nudge = np.array([0.0, 0.0, 0.0])

        ii = 0
        history = ""
        do_leap = False
        while ii < max_trials:
            s1 = s
            s1_thickness = s1.positions[:, 2].max() - s1.positions[:, 2].min()
            if do_leap:
                atom = Atoms(
                    positions=[
                        [
                            s.cell.cellpar()[0] * np.random.random(),
                            s.cell.cellpar()[1] * np.random.random(),
                            nudge[2] + s1_thickness + 2.5 / 2,
                        ]
                    ],
                    pbc=True,
                    cell=s.cell,
                    symbols=[atom_symbol],
                )

            else:
                atom = Atoms(
                    positions=[
                        [
                            position[0] + nudge[0],
                            position[1] + nudge[1],
                            nudge[2] + s1_thickness + 2.5 / 2,
                        ]
                    ],
                    symbols=[atom_symbol],
                    pbc=True,
                    cell=s.cell,
                )

            intercalation = s1 + atom

            intercalation = self.ase_to_pymatgen(intercalation)

            n = intercalation.get_neighbors(intercalation[-1], 3.5)
            n_Li_distances = []
            needs_nudging = False
            if len(n) == 0:
                nudge[2] += -0.01
            else:
                for n_Li in n:
                    n_Li_distances += [
                        self.distance(intercalation[-1].coords, n_Li.coords)
                    ]
                    history += str([n_Li_distances]) + "\n"
                    min_bond, max_bond = self._get_min_max_bonds(
                        intercalation[-1].specie.Z, n_Li.specie.Z
                    )
                    if (
                        self.distance(intercalation[-1].coords, n_Li.coords) < min_bond
                        or self.distance(intercalation[-1].coords, n_Li.coords)
                        > max_bond
                        and not self.is_image(n_Li, intercalation[-1])
                    ):
                        nudge += self._nudgeVector(
                            intercalation[-1].coords, n_Li.coords, min_bond, max_bond
                        )
                        needs_nudging = True
                if not needs_nudging:
                    return OgStructure(intercalation)
                elif ii % 1000 == 0:
                    do_leap = True
                elif ii == max_trials - 1:
                    return False
            ii += 1
        return False

    def relax(
        self, relaxation_method="m3gnet", cellbounds=None, steps=1000, relax_cell=True, fmax=0.05
    ):
        if relaxation_method == "m3gnet":
            relaxer = Relaxer(relax_cell=relax_cell)
            relax_results = relaxer.relax(
                self.structure, verbose=True, steps=steps, fmax=fmax
            )
            self.structure = relax_results["final_structure"]
            self.total_energy = relax_results["trajectory"].energies[-1]
        else:
            raise Exception("Only m3gnet is supported for relaxation at this stage.")

    def generate_neb(
        self, moving_atom_species, num_images=5, r=3, relaxation_method=None
    ) -> None:
        structure = self.structure
        self.neb_paths = []
        for i_site in range(len(structure)):
            if structure[i_site].specie.symbol == moving_atom_species:
                all_neighbors = structure.get_neighbors(site=structure[i_site], r=r)
                neighbors = []
                for site in all_neighbors:
                    if site.specie.symbol == moving_atom_species:
                        neighbors += [site]
                print("og:Checking site", i_site, ": Surrounded by", len(neighbors))
                for neighbor in neighbors:
                    i_neighbor_site = self._get_site_for_neighbor_site(neighbor)
                    print(i_neighbor_site)
                    if i_neighbor_site is None:
                        raise Exception("og:Really? Wrong site in neighbor list!")
                    if [i_site, i_neighbor_site] in self.neb_paths or [
                        i_neighbor_site,
                        i_site,
                    ] in self.neb_paths:
                        continue
                    else:
                        new_structure = structure.copy()
                        self.neb_paths += [[i_site, i_neighbor_site]]
                        neb_folder = (
                            "neb_path_" + str(i_neighbor_site) + "_" + str(i_site)
                        )

                        ogs = OgStructure(new_structure)
                        new_structure = ogs.center(i_site, [0.5, 0.5, 0.5]).structure

                        initial_structure = new_structure.copy()
                        final_structure = new_structure.copy()
                        initial_structure.remove_sites([i_site, i_neighbor_site])
                        initial_structure.append(
                            species=new_structure[i_site].species,
                            coords=new_structure[i_site].frac_coords,
                        )
                        initial = OgStructure.pymatgen_to_ase(initial_structure)
                        final_structure.remove_sites([i_site, i_neighbor_site])
                        final_structure.append(
                            species=new_structure[i_neighbor_site].species,
                            coords=new_structure[i_neighbor_site].frac_coords,
                        )
                        final = OgStructure.pymatgen_to_ase(final_structure)

                        self.images = [initial]
                        self.images += [initial.copy() for i in range(num_images)]
                        self.images += [final]
                        self.neb = NEB(self.images)
                        os.mkdir(neb_folder)
                        self.neb.interpolate(mic=True)
                        for i in range(len(self.images)):
                            self.images[i] = OgStructure(self.images[i])
                            if relaxation_method == "m3gnet":
                                self.images[i] = self.images[i].relax()
                            image_str = self.images[i].structure.to(fmt="poscar")
                            f = open(neb_folder + "/" + str(i).zfill(2), "w")
                            f.write(image_str)
                            f.close()

    def substitutions(self, atom_X, atom_X_substitution, atol=0.001):
        new_structures = unique_structure_substitutions(
            self.structure, atom_X, atom_X_substitution, atol=atol
        )
        if "X" not in atom_X_substitution.keys():
            return [OgStructure(s) for s in new_structures]
        else:
            updated_structures = []
            for s in new_structures:
                s.remove_species(["X"])
                updated_structures += [s]
            return [OgStructure(s) for s in updated_structures]

    def substitutions_random(self, atom_X, atom_X_substitution):
        atom_X_s = []
        for k in atom_X_substitution.keys():
            atom_X_s += atom_X_substitution[k] * [k]

        random.shuffle(atom_X_s)
        for iatom in range(len(self.structure)):
            if self.structure[iatom].specie.symbol == atom_X:
                self.structure.replace(iatom, Element(atom_X_s.pop()))
        return self

    def simulate(
        self,
        thermostat="anderson",
        steps=10000,
        temperature=300,
        ensemble="nvt",
        timestep=1,
        loginterval=1000,
        folder_tag=None,
    ):
        self.dt = timestep
        if not os.path.isdir("og_lab"):
            os.mkdir("og_lab")
        if folder_tag is None:
            self.folder_tag = str(uuid.uuid4())

        self.folder_tag = "simulation_" + self.folder_tag
        os.mkdir("og_lab/" + self.folder_tag)

        self.trajectory_file = (
            "og_lab/" + self.folder_tag + "/" + str(temperature) + ".traj"
        )
        self.log_file = "og_lab/" + self.folder_tag + "/" + str(temperature) + ".log"
        md = MolecularDynamics(
            atoms=self.to_ase(),
            thermostat=thermostat,
            temperature=temperature,
            timestep=timestep,
            ensemble=ensemble,
            trajectory=self.trajectory_file,
            logfile=self.log_file,
            loginterval=loginterval,
        )
        md.run(steps=steps)

    def calculate_diffusivity(
        self, calculation_type="tracer", axis="all", ignore_n_images=0
    ):
        if self.trajectory_file:
            diffusion_coefficients = DiffusionCoefficient(
                Trajectory(self.trajectory_file),
                self.dt * 1e-15,
                calculation_type=calculation_type,
                axis=axis,
            )
            diffusion_coefficients.calculate(ignore_n_images=ignore_n_images)
            self.diffusion_coefficients = (
                diffusion_coefficients.get_diffusion_coefficients()
            )

            # Plotting the MSD curve for each species in the structure

            plt.figure(figsize=(15, 10))
            MSDs = []
            plots = []
            n = len(diffusion_coefficients.timesteps)
            print("og:Plotting MSD using", n, "images")

            for sym_index in range(diffusion_coefficients.no_of_types_of_atoms):
                MSD = np.zeros(len(diffusion_coefficients.timesteps[1:]))
                for xyz in range(3):
                    MSD += diffusion_coefficients.xyz_segment_ensemble_average[0][
                        sym_index
                    ][xyz]
                MSD /= 3
                MSDs += [MSD]
                label = diffusion_coefficients.types_of_atoms[sym_index]
                # Add scatter graph  for the mean square displacement data in this segment
                (l,) = plt.plot(
                    diffusion_coefficients.timesteps[1:], MSD, label=label, linewidth=1
                )
                plots += [l]
            plt.legend(handles=plots)
            plt.ylabel("MSD")
            plt.savefig(
                "og_lab/" + self.folder_tag + "/MSD_" + calculation_type + "_" + axis,
                bbox_inches="tight",
            )
            plt.clf()

            return self.diffusion_coefficients
        else:
            print("og:You have to run a simulation first!")

    def xrd(self, two_theta_range=(0, 180)):
        if self.structure_tag is None:
            tag = self.structure.formula
        else:
            tag = self.structure_tag
        xrd_calculator = XRDCalculator()
        p = xrd_calculator.get_pattern(self.structure, two_theta_range=two_theta_range)
        plt.figure(figsize=(15, 10))
        print("og:Plotting the XRD pattern")
        plt.plot(p.x, p.y, linewidth=1)
        plt.xlabel(r"$2\Theta$")
        plt.xticks(range(two_theta_range[0], two_theta_range[1] + 10, 10))
        plt.ylabel(r"Intensity")
        plt.savefig("og_lab/XRD_" + tag, bbox_inches="tight")
        plt.clf()
        return p

    def get_delta_vector(self, structure):
        """
        Gets the difference between the atomic positions of the current structure and `structure`.
        """
        structure = OgStructure(structure)
        lattice = self.structure.lattice
        return np.vstack(
            [
                pbc_shortest_vectors(
                    lattice, self.structure.frac_coords[i], structure().frac_coords[i]
                )
                for i in range(len(self))
            ]
        ).reshape(len(self), 3)

    def get_rdf(
        self,
        rmax: float,
        nbins: int,
        distance_matrix: Optional[np.ndarray] = None,
        elements: Optional[Union[List[int], Tuple]] = None,
        no_dists: Optional[bool] = False,
        volume: Optional[float] = None,
    ):
        """Returns two numpy arrays; the radial distribution function
        and the corresponding distances of the supplied atoms object.
        If no_dists = True then only the first array is returned.

        Note that the rdf is computed following the standard solid state
        definition which uses the cell volume in the normalization.
        This may or may not be appropriate in cases where one or more
        directions is non-periodic.

        Parameters:

        rmax : float
            The maximum distance that will contribute to the rdf.
            The unit cell should be large enough so that it encloses a
            sphere with radius rmax in the periodic directions.

        nbins : int
            Number of bins to divide the rdf into.

        distance_matrix : numpy.array
            An array of distances between atoms, typically
            obtained by atoms.get_all_distances().
            Default None meaning that it will be calculated.

        elements : list or tuple
            List of two atomic numbers. If elements is not None the partial
            rdf for the supplied elements will be returned.

        no_dists : bool
            If True then the second array with rdf distances will not be returned.

        volume : float or None
            Optionally specify the volume of the system. If specified, the volume
            will be used instead atoms.cell.volume.
        """

        # First check whether the cell is sufficiently large

        def check_cell_and_r_max(atoms: Atoms, rmax: float) -> None:
            cell = atoms.get_cell()
            pbc = atoms.get_pbc()

            vol = atoms.cell.volume

            for i in range(3):
                if pbc[i]:
                    axb = np.cross(cell[(i + 1) % 3, :], cell[(i + 2) % 3, :])
                    h = vol / np.linalg.norm(axb)
                    if h < 2 * rmax:
                        recommended_r_max = get_recommended_r_max(cell, pbc)
                        raise Exception(
                            "The cell is not large enough in "
                            f"direction {i}: {h:.3f} < 2*rmax={2 * rmax: .3f}. "
                            f"Recommended rmax = {recommended_r_max}"
                        )

        def get_recommended_r_max(cell: Cell, pbc: List[bool]) -> float:
            recommended_r_max = 5.0
            vol = cell.volume
            for i in range(3):
                if pbc[i]:
                    axb = np.cross(
                        cell[(i + 1) % 3, :], cell[(i + 2) % 3, :]  # type: ignore
                    )  # type: ignore
                    h = vol / np.linalg.norm(axb)
                    recommended_r_max = min(h / 2 * 0.99, recommended_r_max)
            return recommended_r_max

        def get_containing_cell_length(atoms: Atoms) -> np.ndarray:
            atom2xyz = atoms.get_positions()
            return np.amax(atom2xyz, axis=0) - np.amin(atom2xyz, axis=0) + 2.0

        def get_volume_estimate(atoms: Atoms) -> float:
            return np.prod(get_containing_cell_length(atoms))

        atoms = self.to_ase()
        vol = atoms.cell.volume if volume is None else volume
        if vol < 1.0e-10:
            raise Exception("og:exception:Undefined volume")

        check_cell_and_r_max(atoms, rmax)

        dm = distance_matrix
        if dm is None:
            dm = atoms.get_all_distances(mic=True)

        rdf = np.zeros(nbins + 1)
        dr = float(rmax / nbins)

        indices = np.asarray(np.ceil(dm / dr), dtype=int)
        natoms = len(atoms)

        if elements is None:
            # Coefficients to use for normalization
            phi = natoms / vol
            norm = 2.0 * math.pi * dr * phi * len(atoms)

            indices_triu = np.triu(indices)
            for index in range(nbins + 1):
                rdf[index] = np.count_nonzero(indices_triu == index)

        else:
            i_indices = np.where(atoms.numbers == elements[0])[0]
            phi = len(i_indices) / vol
            norm = 4.0 * math.pi * dr * phi * natoms

            for i in i_indices:
                for j in np.where(atoms.numbers == elements[1])[0]:
                    index = indices[i, j]
                    if index <= nbins:
                        rdf[index] += 1

        rr = np.arange(dr / 2, rmax, dr)
        rdf[1:] /= norm * (rr * rr + (dr * dr / 12))

        if no_dists:
            return rdf[1:]

        return rdf[1:], rr

    def zero_z(self):
        x = self.structure.cart_coords[:, 0]
        y = self.structure.cart_coords[:, 1]
        z = self.structure.cart_coords[:, 2]
        z = z - z.min()
        new_coords = np.array([x, y, z]).T
        self.structure = Structure(
            lattice=self.structure.lattice,
            species=self.structure.species,
            coords=new_coords,
            coords_are_cartesian=True,
        )
        return self

    def repeat(self, scaling_matrix: ArrayLike):
        self.structure = self.structure.make_supercell(scaling_matrix)
        return self
    
    def scale(self, s: Union[list, float]):
        if isinstance(s, list):
            new_lattice = Lattice.from_parameters(
                self.structure.lattice.a * s[0],
                self.structure.lattice.b * s[1],
                self.structure.lattice.c * s[2],
                self.structure.lattice.alpha,
                self.structure.lattice.beta,
                self.structure.lattice.gamma,
            )
        elif isinstance(s, float):
            new_lattice = Lattice.from_parameters(
                self.structure.lattice.a * s,
                self.structure.lattice.b * s,
                self.structure.lattice.c * s,
                self.structure.lattice.alpha,
                self.structure.lattice.beta,
                self.structure.lattice.gamma,
            )
        else:
            raise Exception("Must provide either a list or a float.")
        self.structure = Structure(
            lattice=new_lattice,
            species=self.structure.species,
            coords=self.structure.frac_coords,
            coords_are_cartesian=False,
        )
        return self

    def scale_lattice_only(self, s: Union[list, float]):
        coords = self.structure.cart_coords
        if isinstance(s, list):
            new_lattice = Lattice.from_parameters(
                self.structure.lattice.a * s[0],
                self.structure.lattice.b * s[1],
                self.structure.lattice.c * s[2],
                self.structure.lattice.alpha,
                self.structure.lattice.beta,
                self.structure.lattice.gamma,
            )
        elif isinstance(s, float):
            new_lattice = Lattice.from_parameters(
                self.structure.lattice.a * s,
                self.structure.lattice.b * s,
                self.structure.lattice.c * s,
                self.structure.lattice.alpha,
                self.structure.lattice.beta,
                self.structure.lattice.gamma,
            )
        else:
            raise Exception("Must provide either a list or a float.")
        self.structure = Structure(
            lattice=new_lattice,
            species=self.structure.species,
            coords=coords,
            coords_are_cartesian=True,
        )
        return self


    def create_ripple(
        self,
        axis,
        units,
        strain,
        amplitude=None,
        relax=False,
        steps=10,
        write_intermediate=False,
    ):
        """
        The amplitude will be obtained from a complicated integral equation, given the length or strain.
        For 2D materials with checkness > 1, ripples will require the use of the relaxer to ensure good enough atomic positions.

        """

        if axis == "y":
            self.make_supercell([1, units, 1])
        elif axis == "x":
            self.make_supercell([units, 1, 1])
        z = self.structure.cart_coords[:, 2]
        z = z - z.min()
        if axis == "y":
            original_length = self.structure.lattice.b
        elif axis == "x":
            original_length = self.structure.lattice.a
        target_length = strain * original_length

        def _make_wave(length):
            # The following uses the arc length equation, but it doesn't work:
            #
            # x, A = symbols('x A')
            # def f(A):
            #     return integrate(sqrt(1+A*(cos(2*x*pi/length))), (x, 0, length))
            # if axis == 'y':
            #     solution = solve(f(A)-self.structure.lattice.b,A)
            # elif axis == 'x':
            #     solution = solve(f(A)-self.structure.lattice.a,A)
            # print(solution)
            # amplitude = solution[0]
            # Alternatively, we can use the pythagoras theorem as an approximation

            if axis == "y":
                amplitude = sqrt(
                    (self.structure.lattice.b / 4) ** 2 - (length / 4) ** 2
                )
                new_lattice = Lattice.from_parameters(
                    self.structure.lattice.a,
                    length,
                    self.structure.lattice.c,
                    self.structure.lattice.alpha,
                    self.structure.lattice.beta,
                    self.structure.lattice.gamma,
                )
            elif axis == "x":
                amplitude = sqrt(
                    (self.structure.lattice.a / 4) ** 2 - (length / 4) ** 2
                )
                new_lattice = Lattice.from_parameters(
                    length,
                    self.structure.lattice.b,
                    self.structure.lattice.c,
                    self.structure.lattice.alpha,
                    self.structure.lattice.beta,
                    self.structure.lattice.gamma,
                )

            print("og:Now wave amplitude is", amplitude)

            self.structure = Structure(
                lattice=new_lattice,
                species=self.structure.species,
                coords=self.structure.frac_coords,
                coords_are_cartesian=False,
            )

            x = self.structure.cart_coords[:, 0]
            y = self.structure.cart_coords[:, 1]
            z = self.structure.cart_coords[:, 2]

            if axis == "y":
                for i in range(len(self.structure)):
                    z[i] = z[i] + amplitude * sin(y[i] * 2 * pi / length)
            elif axis == "x":
                for i in range(len(self.structure)):
                    z[i] = z[i] + amplitude * sin(x[i] * 2 * pi / length)
            new_coords = np.array([x, y, z]).T
            self.structure = Structure(
                lattice=new_lattice,
                species=self.structure.species,
                coords=new_coords,
                coords_are_cartesian=True,
            )

        if not relax:
            _make_wave(target_length)
            return self
        else:
            # Thicker layers will require a relaxation loop
            # The wave is created over multiple relaxation steps, each step the amplitude increases by epsilon
            length = original_length
            delta = (original_length - target_length) / steps
            length -= delta

            fn = str(uuid.uuid4())

            print("og:Creating a wave by relaxing the structure with delta =", delta)
            while length >= target_length:
                print(
                    "og:Current wavelength:",
                    length,
                    ", target wavelength:",
                    target_length,
                )
                _make_wave(length)
                self.relax(relax_cell=False)
                if write_intermediate:
                    self.structure.to("ripple_" + fn + '_' + str(length) + ".cif")
                length -= delta

            return self

    def create_helix(self, length, amplitude):
        x = self.structure.cart_coords[:, 0]
        y = self.structure.cart_coords[:, 1]
        z = self.structure.cart_coords[:, 2]

        for i in range(len(self.structure)):
            z[i] = z[i] + amplitude * np.cos(x[i] * 2 * np.pi / length)
            y[i] = y[i] + amplitude * np.sin(x[i] * 2 * np.pi / length)

        new_coords = np.array([x, y, z]).T
        return OgStructure(
            Structure(
                lattice=self.structure.lattice,
                species=self.structure.species,
                coords=new_coords,
                coords_are_cartesian=True,
            )
        )

    """
    public XYZ rotationalAlignmentOnYAxis(int a, int b) {
        int atomA = a - 1;
        int atomB = b - 1;
        double r12X = x[atomA] - x[atomB];
        double r12Y = y[atomA] - y[atomB];
        double r12Z = z[atomA] - z[atomB];
        double thetaXY = Math.acos(r12X / Math.sqrt(r12X * r12X + r12Y * r12Y));
        double thetaXZ = Math.atan(r12Z / Math.sqrt(r12X * r12X + r12Y * r12Y));

        //Perform rotation 1: about z axis
        for (int i = 0; i < N; i++) {
            double oldx = x[i];
            double oldy = y[i];
            x[i] = oldx * Math.cos(thetaXY) - oldy * Math.sin(thetaXY);
            y[i] = oldx * Math.sin(thetaXY) + oldy * Math.cos(thetaXY);
        }

        //Perform rotation 2: about y axis
        for (int i = 0; i < N; i++) {
            double oldx = x[i];
            double oldz = z[i];
            x[i] = oldx * Math.cos(thetaXZ) + oldz * Math.sin(thetaXZ);
            z[i] = -oldx * Math.sin(thetaXZ) + oldz * Math.cos(thetaXZ);

        }

        double thetaYZ = -Math.atan(z[atomA] / y[atomA]);

        //Perform rotation 3: about x axis
        for (int i = 0; i < N; i++) {
            double oldy = y[i];
            double oldz = z[i];
            y[i] = oldy * Math.cos(thetaYZ) - oldz * Math.sin(thetaYZ);
            z[i] = oldy * Math.sin(thetaYZ) + oldz * Math.cos(thetaYZ);
        }
        return this;
    }

    public XYZ rotationalAlignmentOnZAxis(int a, int b) {
        int atomA = a - 1;
        int atomB = b - 1;
        double r12X = Math.abs(x[atomA] - x[atomB]);
        double r12Y = Math.abs(y[atomA] - y[atomB]);
        double r12Z = Math.abs(z[atomA] - z[atomB]);
        double thetaXY = Math.asin(-r12Y / Math.sqrt(r12X * r12X + r12Y * r12Y));
        double thetaXZ = Math.PI / 2 - Math.atan(-r12Z / Math.sqrt(r12X * r12X + r12Y * r12Y));

        //Perform rotation 1: about z axis
        for (int i = 0; i < N; i++) {
            double oldx = x[i];
            double oldy = y[i];
            x[i] = oldx * Math.cos(thetaXY) - oldy * Math.sin(thetaXY);
            y[i] = oldx * Math.sin(thetaXY) + oldy * Math.cos(thetaXY);
        }

        //Perform rotation 2: about y axis
        for (int i = 0; i < N; i++) {
            double oldx = x[i];
            double oldz = z[i];
            x[i] = oldx * Math.cos(thetaXZ) + oldz * Math.sin(thetaXZ);
            z[i] = -oldx * Math.sin(thetaXZ) + oldz * Math.cos(thetaXZ);

        }
        return this;
    }

    public XYZ rotationalAlignmentAlongZAxis(int a, int b) {
        int atomA = a - 1;
        int atomB = b - 1;
        double r12X = (x[atomA] - x[atomB]);
        double r12Y = (y[atomA] - y[atomB]);
        double r12Z = (z[atomA] - z[atomB]);
        double thetaXY = 0, thetaXZ = 0;
        thetaXY = Math.acos(r12X / Math.sqrt(r12X * r12X + r12Y * r12Y));
        thetaXZ = -Math.PI / 2 + Math.atan(r12Z / Math.sqrt(r12X * r12X + r12Y * r12Y));

        //Perform rotation 1: about z axis
        for (int i = 0; i < N; i++) {
            double oldx = x[i];
            double oldy = y[i];
            x[i] = oldx * Math.cos(thetaXY) - oldy * Math.sin(thetaXY);
            y[i] = oldx * Math.sin(thetaXY) + oldy * Math.cos(thetaXY);
        }

        //Perform rotation 2: about y axis
        for (int i = 0; i < N; i++) {
            double oldx = x[i];
            double oldz = z[i];
            x[i] = oldx * Math.cos(thetaXZ) + oldz * Math.sin(thetaXZ);
            z[i] = -oldx * Math.sin(thetaXZ) + oldz * Math.cos(thetaXZ);

        }
        return this;
    }
    public int getNumValenceElectrons() throws InvalidElementException {
        int valence = 0;
        for (int i = 0; i < N; i++) {
            valence += AtomicData.getValence(atom[i]);
        }
        return valence;
    }
    """

    def make_supercell(self, l):
        self.structure = self.structure.make_supercell(l)
        return self

    def add_interstitial(
        self, atom: Union[str, Element], cutoff=4, divisions=[10, 10, 10]
    ):
        """Creates an interstitial defect in the structure in-place by adding the atom to the available voids."""

        def density(r, R):
            return np.exp(-np.dot(r - R, r - R))

        supercell = self.structure.copy()
        supercell.make_supercell([3, 3, 3])
        divisions = np.array(divisions) * 3
        n = np.zeros(divisions)
        for i in range(divisions[0]):
            for j in range(divisions[1]):
                for k in range(divisions[2]):
                    f = [i / divisions[0], j / divisions[1], k / divisions[2]]
                    r = np.dot(f, supercell.lattice.matrix)

                    for R in supercell.cart_coords:
                        if self.distance(R, r) <= cutoff:
                            n[i, j, k] += density(r, R)
        n = n[10:20, 10:20, 10:20]
        p = np.where(n == np.min(n))
        p = [
            p[0][0] / (divisions[0] / 3),
            p[1][0] / (divisions[1] / 3),
            p[2][0] / (divisions[2] / 3),
        ]
        print(p)
        self.structure.append(atom, p)
        return self

    def get_atom_count(self,atom:str):
        an = np.where(np.array(atomic_data.labels) == atom)[0]
        c = np.array(self.structure.atomic_numbers)
        return len(c[c == an])
    
    def calculate_molecular_mass(self):
        m = 0
        for a in self.structure:
            m += a.specie.atomic_mass
        return m

    def get_graph(self):
        """Converts the structure into a graph using the M3GNET tensorflow implementation"""
        r = RadiusCutoffGraphConverter()
        mg = r.convert(self.structure)
        graph = tf_compute_distance_angle(mg.as_list())
        return graph

    def calculate_theoretical_capacity(self,charge_carrier:str,n=None):
        '''
        Formula:
        Q = (nF) / (3600*M_w)*1000   mAh g-1
        '''
        if n is None:
            n = self.get_atom_count(charge_carrier)
        M_w = self.calculate_molecular_mass()/1000
        return n*F/(3600*M_w)