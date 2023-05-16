<img src="./assets/logo.svg" width="200px">

# Oganesson

`oganesson` (`og` for short) is a python package that enables you to apply artificial intelligence workflows to your material discovery projects.


## Installation

You can install `og` using the `pip` command as follows:

`pip install oganesson`

# Features

`og` is currently under active development. The following features are currently available.


## Machine learning descriptors

`og` will bring together machine learning descriptors for materials and molecules. At the moment, the three descriptors available are those introduced in this [publication](https://doi.org/10.1186/s13321-022-00658-9). 

Each descriptor has its own class, which extends the `Descriptors` class in the `oganesson.descriptors` module. Here is an example of how to describe a structure using the `BACD` and `SymmetryFunctions` descriptor classes.

```python
from oganesson.descriptors.bacd import BACD
from oganesson.descriptors.symmetry_functions import SymmetryFunctions
from oganesson.ogstructure import OgStructure

bacd = BACD(OgStructure(file_name='examples/structures/mp-541001_LiInI4.cif'))
print(bacd.describe())

sf = SymmetryFunctions(OgStructure(file_name='examples/structures/mp-541001_LiInI4.cif'))
print(sf.describe())
```

## Genetic algorithms

The main purpose of `og` is to make complex artificial intelligence workflows easy to compose. Here is an example: running a genetic search for structures, where the structure optimization is performed using the M3GNET neural network model.

```python
from oganesson.genetic_algorithms import GA
ga = GA(['Na']*4 + ['H']*4)
for i in range(10):
    ga.evolve()
```

## Generation of the diffusion path for NEB calculations

The most painful part of doing transition state calculations in VASP is in building the images. The following code makes this happen in 2 lines of code. You only need to specify the structure file, and the atomic species you want to diffuse, and OgStructure will generate a folder for each path, and writes the POSCAR image files in each of these folders.

```python
from oganesson.ogstructure import OgStructure
og = OgStructure(file_name='examples/structures/mp-541001_LiInI4.cif')
og.generate_neb('Li')
```