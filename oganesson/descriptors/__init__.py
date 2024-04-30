import importlib as imp
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
    imp.util.find_spec('gpaw')
    from oganesson.descriptors.rosa import _ROSA
    ROSA = _ROSA
except ImportError:
    print('GPAW is not installed, and therefore you cannot use the ROSA descriptors.')

try:
    imp.util.find_spec('dscribe')
    from oganesson.descriptors.dscribe import _DScribeACSF, _DScribeSOAP, _DScribeCoulombMatrix, _DScribeEwaldSumMatrix, _DScribeSineMatrix
    DscribeACSF = _DScribeACSF
    DScribeSOAP = _DScribeSOAP
    DScribeCoulombMatrix = _DScribeCoulombMatrix
    DScribeEwaldSumMatrix = _DScribeEwaldSumMatrix
    DScribeSineMatrix = _DScribeSineMatrix
except ImportError:
    print('DScribe is not installed, and therefore you cannot use the DScribe descriptors.')


class DescriptorsName(Enum):
    BACD = 0
    ROSA = 1
    SymmetryFunctions = 2
    DscribeACSF = 3


class Describe:
    @staticmethod
    def describe(structure: Union[Atoms, Structure, str, OgStructure],
                 descriptor: Descriptors,
                 string_format: str = None) -> None:
        # if not isinstance(descriptor, Descriptors):
        #     raise Exception('The descriptor argument must be of type Descriptors.')
        descriptor_object = descriptor(structure)
        return descriptor_object.describe()
