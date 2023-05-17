import imp
from oganesson.ogai import OgAI
from oganesson.descriptors import BACD
from oganesson.descriptors import SymmetryFunctions
from oganesson.ogstructure import OgStructure
from oganesson.genetic_algorithms import GA

__all__ = ("OgAI",
           "BACD",
           "SymmetryFunctions",
           "OgStructure",
           "GA")

try:
    imp.find_module('gpaw')
    from oganesson.descriptors import ROSA
    __all__ += tuple("ROSA")
except ImportError:
    print('GPAW is not installed, and therefore you cannot use the ROSA descriptors.')
