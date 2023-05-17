import imp
from enum import Enum
from ase import Atoms
from pymatgen.core import Structure
from oganesson.ogstructure import OgStructure
from oganesson.descriptors.descriptors import Descriptors
from oganesson.descriptors.bacd import _BACD
from oganesson.descriptors.symmetry_functions import _SymmetryFunctions
from typing import Union

BACD = _BACD
SymmetryFunctions = _SymmetryFunctions

try:
    imp.find_module('gpaw')
    from oganesson.descriptors.rosa import _ROSA
    ROSA = _ROSA
except ImportError:
    print('GPAW is not installed, and therefore you cannot use the ROSA descriptors.')


class DescriptorsName(Enum):
    BACD = 0
    ROSA = 1
    SymmetryFunctions = 2


class Describe:
    def __init__(self,
                 structure: Union[Atoms, Structure, str, OgStructure],
                 descriptor: Union[DescriptorsName, Descriptors],
                 string_format: str = None) -> None:
        pass
