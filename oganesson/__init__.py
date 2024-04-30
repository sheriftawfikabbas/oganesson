import importlib as imp
from oganesson.ogai import OgAI
from oganesson.descriptors import BACD
from oganesson.descriptors import SymmetryFunctions
from oganesson.ogstructure import OgStructure
from oganesson.genetic_algorithms import GA

__all__ = ["OgAI",
           "BACD",
           "SymmetryFunctions",
           "OgStructure",
           "GA"]

try:
    imp.util.find_spec('gpaw')
    from oganesson.descriptors import ROSA
    __all__ += ["ROSA"]
except ImportError:
    pass

try:
    imp.util.find_spec('dscribe')
    from oganesson.descriptors import DScribeACSF, DScribeSOAP, DScribeCoulombMatrix, DScribeEwaldSumMatrix, DScribeSineMatrix
    __all__ += ["DScribeACSF",
                "DScribeSOAP",
                "DScribeCoulombMatrix",
                "DScribeEwaldSumMatrix",
                "DScribeSineMatrix"]
except ImportError:
    pass

__all__ = tuple(__all__)
