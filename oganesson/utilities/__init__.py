
import numpy as np
from ase.atoms import Atoms, Cell
from pymatgen.core import Structure
from ase.io import read, write
from ase.db import connect
from ase.atoms import Atoms
import oganesson.utilities.atomic_data


def formula(structure):
    Data_form = []
    for i in range(len(structure)):
        A = structure[i]
        Data_form.append(A.as_dict()["species"][0]["element"])
    return Data_form

def get_index_positions(list_of_elems, element):
    return np.where(np.array(list_of_elems) == element)

def epsilon(a, b):
    if abs(a-b) < 1e-6:
        return True
    else:
        return False

