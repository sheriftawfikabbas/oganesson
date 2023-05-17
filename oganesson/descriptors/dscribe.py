import numpy as np
from oganesson.ogstructure import OgStructure
from pymatgen.core import Structure
from ase import Atoms
from typing import Union
from oganesson.descriptors import Descriptors
from dscribe.descriptors import ACSF,\
    LMBTR,\
    SineMatrix,\
    SOAP,\
    ValleOganov,\
    CoulombMatrix,\
    EwaldSumMatrix


class DScribeDescriptors(Descriptors):
    def __init__(self, structure: Union[Atoms, Structure, str, OgStructure]) -> None:
        super().__init__(structure)
        self.dscribe = None

    def describe(self):
        original = self.dscribe.create(self.structure.to_ase())
        if len(original.shape) == 2:
            mean = [original[:, i].mean() for i in range(original.shape[1])]
            min = [original[:, i].min() for i in range(original.shape[1])]
            max = [original[:, i].max() for i in range(original.shape[1])]
            std = [original[:, i].std() for i in range(original.shape[1])]
            return mean + min + max + std
        else:
            return original


class _DScribeSineMatrix(DScribeDescriptors):
    def __init__(self, structure: Union[Atoms, Structure, str, OgStructure],
                 n_atoms_max=None, permutation='sorted_l2', sigma=None, seed=None, sparse=False) -> None:
        super().__init__(structure)
        if n_atoms_max is None:
            self.n_atoms_max = len(self.structure())
        else:
            self.n_atoms_max = n_atoms_max
        self.permutation = permutation
        self.sigma = sigma
        self.seed = seed
        self.sparse = sparse
        print('Number of atoms:', self.n_atoms_max)
        self.dscribe = SineMatrix(
            self.n_atoms_max, permutation, sigma, seed, sparse)


class _DScribeEwaldSumMatrix(DScribeDescriptors):
    def __init__(self, structure: Union[Atoms, Structure, str, OgStructure],
                 n_atoms_max=10, permutation='sorted_l2', sigma=None, seed=None, sparse=False) -> None:
        super().__init__(structure)
        if n_atoms_max is None:
            self.n_atoms_max = len(self.structure())
        else:
            self.n_atoms_max = n_atoms_max
        self.permutation = permutation
        self.sigma = sigma
        self.seed = seed
        self.sparse = sparse
        self.dscribe = EwaldSumMatrix(
            self.n_atoms_max, permutation, sigma, seed, sparse)


class _DScribeCoulombMatrix(DScribeDescriptors):
    def __init__(self, structure: Union[Atoms, Structure, str, OgStructure],
                 n_atoms_max=10, permutation='sorted_l2', sigma=None, seed=None, sparse=False) -> None:
        super().__init__(structure)
        if n_atoms_max is None:
            self.n_atoms_max = len(self.structure())
        else:
            self.n_atoms_max = n_atoms_max
        self.permutation = permutation
        self.sigma = sigma
        self.seed = seed
        self.sparse = sparse
        self.dscribe = CoulombMatrix(
            self.n_atoms_max, permutation, sigma, seed, sparse)


class _DScribeACSF(DScribeDescriptors):
    def __init__(self, structure: Union[Atoms, Structure, str, OgStructure],
                 g2_params=None, g3_params=None, g4_params=None, g5_params=None,
                 rcut=6.0,
                 periodic: bool = False,
                 sparse: bool = False) -> None:
        super().__init__(structure)
        self.g2_params = g2_params
        self.g3_params = g3_params
        self.g4_params = g4_params
        self.g5_params = g5_params
        self.rcut = rcut
        self.periodic = periodic
        self.sparse = sparse
        self.dscribe = ACSF(
            species=self.structure().symbol_set,
            rcut=6.0,
            g2_params=[[1, 1], [1, 2], [1, 3]
                       ] if g2_params is None else g2_params,
            g3_params=g3_params,
            g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1],
                       [1, 2, -1]]if g4_params is None else g4_params,
            g5_params=g5_params
        )


class _DScribeSOAP(DScribeDescriptors):
    def __init__(self, structure: Union[Atoms, Structure, str, OgStructure],
                 rcut=6,
                 nmax=6,
                 lmax=6,
                 sigma=1.0,
                 rbf="gto",
                 weighting=None,
                 crossover: bool = True,
                 average="off",
                 periodic: bool = False,
                 sparse: bool = False,
                 ) -> None:
        super().__init__(structure)
        self.rcut = rcut,
        self.nmax = nmax,
        self.lmax = lmax,
        self.sigma = sigma,
        self.rbf = rbf,
        self.weighting = weighting,
        self.crossover = crossover,
        self.average = average,
        self.periodic = periodic,
        self.sparse = sparse

        self.dscribe = SOAP(
            species=self.structure().symbol_set,
            rcut=rcut,
            nmax=nmax,
            lmax=lmax,
            sigma=sigma,
            rbf=rbf,
            weighting=weighting,
            crossover=crossover,
            average=average,
            periodic=periodic,
            sparse=sparse
        )
