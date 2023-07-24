from typing import Union
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from urllib.request import urlopen
import pandas as pd
import numpy as np
from oganesson.descriptors.descriptors import Descriptors
from oganesson.ogstructure import OgStructure
from typing import Union

class _BACD(Descriptors):
    def __init__(self, structure: Union[Atoms, Structure, str, OgStructure]) -> None:
        super().__init__(structure)

    def describe(self):
        
        
        
        descriptors_list = []
        return descriptors_list
