from ase.io import write
from ase.ga import get_raw_score
from ase.ga.data import DataConnection
from ase.ga.population import Population
from ase.ga.utilities import closest_distances_generator, CellBounds
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.offspring_creator import OperationSelector
from ase.ga.standardmutations import StrainMutation
from ase.ga.soft_mutation import SoftMutation
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase import Atoms
from ase.data import atomic_numbers
from ase.ga.startgenerator import StartGenerator
from ase.ga.data import PrepareDB
from ase.ga import set_raw_score
from m3gnet.models import Relaxer
from oganesson.ogstructure import OgStructure
import os
from typing import List


class GA:
    def finalize(self, atoms, energy):
        raw_score = -energy
        set_raw_score(atoms, raw_score)

    def relax(self, atoms, cellbounds=None):
        relaxer = Relaxer()
        structure = OgStructure.ase_to_pymatgen(atoms)
        relax_results = relaxer.relax(structure, verbose=True)

        e = float(relax_results['trajectory'].energies[-1])
        self.finalize(atoms, energy=e)
        return OgStructure.pymatgen_to_ase(relax_results['final_structure'])

    def __init__(self, population: List[OgStructure]=None, species=None, population_size=20, box_volume=240,
                 a: List = [3, 10], b: List = [3, 10], c: List = [3, 10],
                 phi: List = [35, 145], chi: List = [35, 145], psi: List = [35, 145]) -> None:
        # Either establish a new population from scratch by randomly filling boxes,
        # or start from a specified population of structures
        self.path = 'og_lab/'
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        import uuid
        self.path = 'og_lab/ga_'+uuid.uuid4().hex
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

        self.path_relaxed = self.path + '/relaxed/'
        if not os.path.isdir(self.path_relaxed):
            os.mkdir(self.path_relaxed)

        print('Starting structural optimization using genetic algorithms..')
        if population is None:
            self.N = population_size
            self.volume = box_volume
            self.species = species
            self.Z = [atomic_numbers[x] for x in self.species]

            self.blmin = closest_distances_generator(atom_numbers=self.Z,
                                                    ratio_of_covalent_radii=0.6)
            self.cellbounds = CellBounds(bounds={'phi': phi, 'chi': chi,
                                                'psi': psi, 'a': a,
                                                'b': b, 'c': c})

            self.splits = {(2,): 1, (1,): 1}
            self.slab = Atoms('', pbc=True)

            self.sg = StartGenerator(self.slab, self.species, self.blmin, box_volume=self.volume,
                                number_of_variable_cell_vectors=3,
                                cellbounds=self.cellbounds, splits=self.splits)

            # Create the database
            self.database = PrepareDB(db_file_name=self.path+'/ga.db',
                                    stoichiometry=self.Z)

            # Generate N random structures
            # and add them to the database
            for i in range(self.N):
                a = self.sg.get_new_candidate()
                self.database.add_unrelaxed_candidate(a)

            # Connect to the database and retrieve some information
            self.database_connection = DataConnection(self.path+'/ga.db')
            self.slab = self.database_connection.get_slab()
            atom_numbers_to_optimize = self.database_connection.get_atom_numbers_to_optimize()
            self.n_top = len(atom_numbers_to_optimize)
                
        elif isinstance(population, list):
            # First, ensure that all structures have their symbols ordered.
            # This is required by ASE's GA.

            self.N = len(population)
            for i in range(self.N):
                population[i] = population[i].sort_species()
            self.species = [x.symbol for x in population[0].structure.species]
            self.Z = population[0].structure.atomic_numbers
            a_values = []
            b_values = []
            c_values = []
            volume_values = []
            alpha_values = []
            beta_values = []
            gamma_values = []
            for p in population:
                a_values += [p.structure.lattice.a]
                b_values += [p.structure.lattice.b]
                c_values += [p.structure.lattice.c]
                alpha_values += [p.structure.lattice.alpha]
                beta_values += [p.structure.lattice.beta]
                gamma_values += [p.structure.lattice.gamma]
                volume_values += [p.structure.volume]
            
            self.volume = max(volume_values)*1.1
            self.blmin = closest_distances_generator(atom_numbers=self.Z,
                                                    ratio_of_covalent_radii=0.6)
            self.cellbounds = CellBounds(bounds={'alpha': [min(alpha_values)*0.9,max(alpha_values)*1.1], 
                                                 'beta': [min(beta_values)*0.9,max(beta_values)*1.1],
                                                'gamma': [min(gamma_values)*0.9,max(gamma_values)*1.1], 
                                                'a':  [min(a_values)*0.9,max(a_values)*1.1],
                                                'b':  [min(b_values)*0.9,max(b_values)*1.1], 
                                                'c':  [min(c_values)*0.9,max(c_values)*1.1]})

            self.splits = {(2,): 1, (1,): 1}
            self.database = PrepareDB(db_file_name=self.path+'/ga.db',
                                    stoichiometry=self.Z)
            for i in range(self.N):
                self.database.add_unrelaxed_candidate(population[i].to_ase())
            
            self.database_connection = DataConnection(self.path+'/ga.db')
            self.slab = self.database_connection.get_slab()
            atom_numbers_to_optimize = self.database_connection.get_atom_numbers_to_optimize()
            self.n_top = len(atom_numbers_to_optimize)
        
        else:
            raise Exception('Wrong input arguments!')

        print('Population size:', self.N)
        self.comp = OFPComparator(n_top=self.n_top, dE=1.0,
                                  cos_dist_max=1e-3, rcut=10., binwidth=0.05,
                                  pbc=[True, True, True], sigma=0.05, nsigma=4,
                                  recalculate=False)

        self.pairing = CutAndSplicePairing(self.slab, self.n_top, self.blmin, p1=1., p2=0., minfrac=0.15,
                                           number_of_variable_cell_vectors=3,
                                           cellbounds=self.cellbounds, use_tags=False)

        self.strainmut = StrainMutation(self.blmin, stddev=0.7, cellbounds=self.cellbounds,
                                        number_of_variable_cell_vectors=3,
                                        use_tags=False)
        self.blmin_soft = closest_distances_generator(
            atom_numbers_to_optimize, 0.1)
        self.softmut = SoftMutation(self.blmin_soft, bounds=[
                                    2., 5.], use_tags=False)
        self.operators = OperationSelector([4., 3., 3.],
                                           [self.pairing, self.softmut, self.strainmut])
        
        # Relax the initial candidates
        while self.database_connection.get_number_of_unrelaxed_candidates() > 0:
            a = self.database_connection.get_an_unrelaxed_candidate()

            relaxed_a = self.relax(a, cellbounds=self.cellbounds)
            a.positions = relaxed_a.positions
            a.cell = relaxed_a.cell
            a.pbc = True
            self.database_connection.add_relaxed_step(a)

            cell = a.get_cell()
            if not self.cellbounds.is_within_bounds(cell):
                self.database_connection.kill_candidate(a.info['confid'])
            else:
                relaxed_a.write(self.path_relaxed+str(a.info['confid'])+'.cif', 'cif')

    def evolve(self, num_offsprings=20):
        self.population = Population(data_connection=self.database_connection,
                                     population_size=self.N,
                                     comparator=self.comp,
                                     logfile=self.path+'/log.txt',
                                     use_extinct=True)

        current_pop = self.population.get_current_population()
        self.strainmut.update_scaling_volume(
            current_pop, w_adapt=0.5, n_adapt=4)
        self.pairing.update_scaling_volume(current_pop, w_adapt=0.5, n_adapt=4)

        for step in range(num_offsprings):
            print('Now starting configuration number {0}'.format(step))

            a3 = None
            while a3 is None:
                a1, a2 = self.population.get_two_candidates()
                a3, desc = self.operators.get_new_individual([a1, a2])

            # Save the unrelaxed candidate
            self.database_connection.add_unrelaxed_candidate(
                a3, description=desc)

            # Relax the new candidate and save it
            relaxed_a = self.relax(a3, cellbounds=self.cellbounds)
            a3.positions = relaxed_a.positions
            a3.cell = relaxed_a.cell
            a3.pbc = True
            self.database_connection.add_relaxed_step(a3)

            # If the relaxation has changed the cell parameters
            # beyond the bounds we disregard it in the population
            cell = a3.get_cell()
            if not self.cellbounds.is_within_bounds(cell):
                self.database_connection.kill_candidate(a3.info['confid'])
            else:
                a3.write(self.path_relaxed+str(a3.info['confid'])+'.cif', 'cif')

            # Update the population
            self.population.update()

            if step % 10 == 0:
                current_pop = self.population.get_current_population()
                self.strainmut.update_scaling_volume(current_pop, w_adapt=0.5,
                                                     n_adapt=4)
                self.pairing.update_scaling_volume(
                    current_pop, w_adapt=0.5, n_adapt=4)
                write(self.path+'/current_population.traj', current_pop)

        print('GA finished after step %d' % step)
        hiscore = get_raw_score(current_pop[0])
        print('Highest raw score = %8.4f eV' % hiscore)

        all_candidates = self.database_connection.get_all_relaxed_candidates()
        write(self.path+'/all_candidates.traj', all_candidates)

        current_pop = self.population.get_current_population()
        write(self.path+'/current_population.traj', current_pop)
