from ase import Atoms
from pymatgen.core import Structure
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from ase.build import niggli_reduce
from gpaw import GPAW
import glob
from oganesson.descriptors.descriptors import Descriptors
from oganesson.ogstructure import OgStructure
from typing import Union

class _ROSA(Descriptors):
    def __init__(self, structure: Union[Atoms, Structure, str, OgStructure], psfolder: str, descriptor_size=100, calculation_type='bulk') -> None:
        super().__init__(structure)
        self.psfolder = psfolder
        self.descriptor_size = descriptor_size
        self.calculation_type = calculation_type
        g = glob.glob(
            self.psfolder + '/*.PBE.gz')

        self.available_atoms = []
        for i in g:
            self.available_atoms += [i.replace(
                psfolder + '/', '').split('.')[0]]

    def fix_psuedo(self, a):
        for i in range(len(a)):
            if not a[i].symbol in self.available_atoms:
                print(a[i].symbol, 'is not available in GPAW, replacing it with Y')
                a[i].symbol = 'Y'

    def get_descriptors_for_structure(self, a):
        half_descriptor_size = int(self.descriptor_size/2)
        calc = GPAW(mode='lcao',
                    xc='PBE',
                    maxiter=1,
                    convergence={'density': 1},
                    kpts=[1, 1, 1])

        # a.set_calculator(calc)
        calc.calculate(a)
        H = calc.hamiltonian
        descriptors_below = []
        descriptors_above = []

        num_atoms = a.get_global_number_of_atoms()

        ev = pd.DataFrame(calc.get_eigenvalues())
        ef = calc.get_fermi_level()
        ev_below = ev.loc[ev.values <= ef]
        ev_above = ev.loc[ev.values > ef]

        for e in ev_below.values.tolist()[::-1]:
            descriptors_below += e
        if len(ev_below) > half_descriptor_size:
            descriptors_below = descriptors_below[0:half_descriptor_size]
        elif len(ev_below) < half_descriptor_size:
            for e in range(half_descriptor_size-len(ev_below)):
                descriptors_below += [0]
        descriptors_below = descriptors_below[::-1]

        for e in ev_above.values.tolist():
            descriptors_above += e
        if len(ev_above) > half_descriptor_size:
            descriptors_above = descriptors_above[0:half_descriptor_size]
        elif len(ev_above) < half_descriptor_size:
            for e in range(half_descriptor_size-len(ev_above)):
                descriptors_above += [0]

        return descriptors_below + descriptors_above + \
            [ef,
             H.e_band/num_atoms,
             H.e_coulomb/num_atoms,
             H.e_entropy/num_atoms,
             H.e_external/num_atoms,
             H.e_kinetic/num_atoms,
             H.e_kinetic0/num_atoms,
             H.e_xc/num_atoms,
             H.e_total_free/num_atoms]

    def describe(self):

        structure = self.structure()

        a = Atoms(pbc=True, cell=structure.lattice.matrix,
                  positions=structure.cart_coords, numbers=structure.atomic_numbers)

        d_pristine = []
        try:
            if self.calculation_type == 'bulk':
                niggli_reduce(a)
            self.fix_psuedo(a)
            d_pristine = self.get_descriptors_for_structure(a)

        except Exception as e:
            print('Problem in GPAW')
        if len(d_pristine) > 0:
            return d_pristine
        else:
            return []
