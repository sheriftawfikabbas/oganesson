## Changelog


### 0.1.21 - 2023-07-14

* Enabled more logging in GA

### 0.1.20 - 2023-07-12

* Hot fix: included get_rdf within OgStructure because it is missing in the most recent version of ASE. Also would benefit from some optimization.

### 0.1.9 - 2023-07-12

* Over-ridden the particle mutation of ASE
* Few fixes

### 0.1.8 - 2023-07-07

* Added new GA operators
* Critical fix for vasp.py
* Several fixes

### 0.1.7 - 2023-06-13

* Creating the GA object won't relax the initial population. The first call for evolve() will do
* Initial GA structures can be written to CIF files

### 0.1.6 - 2023-06-13

* GA population can be provided by user
* Added random atom substitution to OgStructure
* Added XRD to OgStructure via pymatgen's API

### 0.1.5 - 2023-06-08

* Added MD to OgStructure
* Added more examples

### 0.1.4 - 2023-05-18

* Added DScribe descriptors
* Avoid importing gpaw/dscribe if they were not installed properly

### 0.1.3 - 2023-05-17

* Fixing setup.py

### 0.1.2 - 2023-05-17

* Fixing setup.py
* GA: Enabling writing structure into a uuid-based folder

### 0.1.1 - 2023-05-16

* Addition of the OgStructure class
* Addition of the ML descriptors BACD, ROSA and SymmetryFunctions

### 0.1.0 - 2023-05-09

* First release
