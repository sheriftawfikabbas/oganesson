from ase import Atoms
from pymatgen.core import Structure
from abc import ABC, abstractmethod
from oganesson.ogstructure import OgStructure
from typing import Union


class Descriptors(ABC):
    def __init__(self, structure: Union[Atoms, Structure, str, OgStructure]) -> None:
        if isinstance(structure, OgStructure):
            self.structure = structure
        else:
            self.structure = OgStructure(structure)

    @abstractmethod
    def describe(self):
        '''
        '''

    def describe_batch(self, directory: str):
        '''
        Transform structures in the directory and load into the destination
        '''

    def sensitivity_analysis(self):
        '''
        Perform standardized tests on the values of descriptors by perturbing the original structure
        and reporting the resulting perturbation in descriptors.
        '''
        

    def is_invariant(self):
        '''
        Tells whether the descriptors are invariant with respect to translation and rotation
        '''
        import numpy as np
        import random
        original_descriptors = np.array(self.describe())
        translated_structure = self.structure().copy()
        translated_structure.translate_sites(range(len(self.structure())),[random.random(), random.random(), random.random()])
        translated_descriptors = self.__class__(translated_structure)
        translated_descriptors = np.array(translated_descriptors.describe())
        if max(original_descriptors-translated_descriptors) > 1e-4:
            return False
        return True
