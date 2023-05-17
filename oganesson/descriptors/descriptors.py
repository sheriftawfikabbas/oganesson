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

    def describe_batch(self,directory:str):
        '''
        Transform structures in the directory and load into the destination
        '''
    
    def sensitivity_analysis(self):
        '''
        Perform standardized tests on the values of descriptors by perturbing the original structure
        and reporting the resulting perturbation in descriptors.
        '''
