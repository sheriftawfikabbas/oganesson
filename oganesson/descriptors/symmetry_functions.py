from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
import numpy as np
from oganesson.descriptors import Descriptors
from oganesson.ogstructure import OgStructure

class SymmetryFunctions(Descriptors):
    def __init__(self, structure: Atoms | Structure | str | OgStructure) -> None:
        super().__init__(structure)

    @staticmethod
    def fc(Rij, iRc, fRc):
        if Rij <= fRc and Rij > iRc:
            return 1
        else:
            return 0


    def G1(self,i, eta, Rs, iRc, fRc):
        sum = 0
        neighbors = self.structure().get_neighbors(site=self.structure()[i], r=fRc)
        ai = self.structure()[i].species.elements[0].symbol
        for j in range(len(neighbors)):
            aj = neighbors[j].species.elements[0].symbol
            Rij = self.structure.distance(self.structure()[i].coords, neighbors[j].coords)
            sum += np.exp(-eta*(Rij-Rs)**2)*self.fc(Rij, iRc, fRc)
        return sum/len(self.structure())


    def G2(self,i, eta, zeta, Rs, iRc, fRc):
        sum = 0
        neighbors = self.structure().get_neighbors(site=self.structure()[i], r=fRc)
        for j in range(len(neighbors)):
            Rij = self.structure.distance(self.structure()[i].coords, neighbors[j].coords)
            sum += np.exp(-eta*(Rij-Rs)**2)*self.fc(Rij, iRc, fRc)*np.exp(-zeta *
                                                                    np.abs(neighbors[j].species.elements[0].Z-self.structure()[i].species.elements[0].Z))
        return sum/len(self.structure())


    def describe(self):
        G1_parameters = {"eta": [1, 2, 3, 4, 5, 6], "Rs": [0, 1, 2, 3, 4], "Rc": [
            [0., 2], [2, 3], [3, 4], [4, 6]]}

        G2_parameters = {"eta": [1, 2, 3, 4, 5, 6], "zeta": [1, 2, 3, 4], "Rs": [0, 1, 2, 3, 4], "Rc": [
            [0., 2], [2, 3], [3, 4], [4, 6]]}

        G1_descriptors = []
        G2_descriptors = []

        for eta in G1_parameters["eta"]:
            for Rs in G1_parameters["Rs"]:
                for Rc in G1_parameters["Rc"]:
                    G = 0
                    for i in range(len(self.structure())):
                        G += self.G1(i, eta, Rs, Rc[0], Rc[1])
                    G1_descriptors += [G]

        for eta in G2_parameters["eta"]:
            for zeta in G2_parameters["zeta"]:
                for Rs in G2_parameters["Rs"]:
                    for Rc in G2_parameters["Rc"]:
                        G = 0
                        for i in range(len(self.structure())):
                            G += self.G2(i, eta, zeta, Rs, Rc[0], Rc[1])
                        G2_descriptors += [G]

        descriptors_list = G1_descriptors + G2_descriptors
        return descriptors_list
