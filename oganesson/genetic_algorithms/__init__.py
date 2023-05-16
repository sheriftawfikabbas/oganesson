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
from ase.ga.utilities import closest_distances_generator, CellBounds
from ase.ga.startgenerator import StartGenerator
from ase.ga.data import PrepareDB
from ase.build import niggli_reduce
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import FIRE
from ase.constraints import ExpCellFilter
from ase.ga import set_raw_score
from m3gnet.models import Relaxer
from oganesson.ogstructure import OgStructure
import os


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

    def __init__(self, blocks, population_size=20, box_volume=240,
                 a: list = [3, 10], b: list = [3, 10], c: list = [3, 10],
                 phi: list = [35, 145], chi: list = [35, 145], psi: list = [35, 145]) -> None:
        self.N = population_size
        self.volume = box_volume
        self.blocks = blocks
        self.Z = [atomic_numbers[x] for x in self.blocks]

        self.blmin = closest_distances_generator(atom_numbers=self.Z,
                                                 ratio_of_covalent_radii=0.6)
        self.cellbounds = CellBounds(bounds={'phi': phi, 'chi': chi,
                                             'psi': psi, 'a': a,
                                             'b': b, 'c': c})

        self.splits = {(2,): 1, (1,): 1}
        self.slab = Atoms('', pbc=True)

        self.sg = StartGenerator(self.slab, self.blocks, self.blmin, box_volume=self.volume,
                                 number_of_variable_cell_vectors=3,
                                 cellbounds=self.cellbounds, splits=self.splits)

        # Create the database
        self.database = PrepareDB(db_file_name='gadb.db',
                                  stoichiometry=self.Z)

        # Generate N random structures
        # and add them to the database
        for i in range(self.N):
            a = self.sg.get_new_candidate()
            self.database.add_unrelaxed_candidate(a)

        # Connect to the database and retrieve some information
        self.database_connection = DataConnection('gadb.db')
        self.slab = self.database_connection.get_slab()
        atom_numbers_to_optimize = self.database_connection.get_atom_numbers_to_optimize()
        self.n_top = len(atom_numbers_to_optimize)

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

        self.path = 'relaxed/'
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

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
                relaxed_a.write(self.path+str(a.info['confid'])+'.cif', 'cif')

    def evolve(self, num_offsprings=20):
        self.population = Population(data_connection=self.database_connection,
                                     population_size=self.N,
                                     comparator=self.comp,
                                     logfile='log.txt',
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
                a3.write(self.path+str(a3.info['confid'])+'.cif', 'cif')

            # Update the population
            self.population.update()

            if step % 10 == 0:
                current_pop = self.population.get_current_population()
                self.strainmut.update_scaling_volume(current_pop, w_adapt=0.5,
                                                     n_adapt=4)
                self.pairing.update_scaling_volume(
                    current_pop, w_adapt=0.5, n_adapt=4)
                write('current_population.traj', current_pop)

        print('GA finished after step %d' % step)
        hiscore = get_raw_score(current_pop[0])
        print('Highest raw score = %8.4f eV' % hiscore)

        all_candidates = self.database_connection.get_all_relaxed_candidates()
        write('all_candidates.traj', all_candidates)

        current_pop = self.population.get_current_population()
        write('current_population.traj', current_pop)
