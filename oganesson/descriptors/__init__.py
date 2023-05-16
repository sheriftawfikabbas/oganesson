from enum import Enum
from ase import Atoms
from pymatgen.core import Structure
from abc import ABC, abstractmethod
from oganesson.ogstructure import OgStructure

class Descriptors(ABC):
    def __init__(self, structure: Atoms | Structure | str | OgStructure) -> None:
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

class DescriptorsName(Enum):
    BACD = 0
    ROSA = 1
    SymmetryFunctions = 2


class Describe:
    def __init__(self, 
                 structure: Atoms | Structure | str | OgStructure, 
                 descriptor: DescriptorsName | Descriptors, 
                 string_format: str = None) -> None:
        pass
