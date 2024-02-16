## Changelog

### 0.1.26 - 2024-02-16

* Various fixes
* Added the get_graph() method to obtain the structure's graph using m3gnet implementation
* Added get_bonds_blocks()
* Fix for SOAP parameters in DScribe
* improved the tutorial notebook

### 0.1.25 - 2023-11-30

* Adding interstitial atoms to a structure
* Several minor fixes

### 0.1.24 - 2023-07-27

* Hot fix for the delta vector function

### 0.1.23 - 2023-07-27

* Addition of delta vector function

### 0.1.22 - 2023-07-16

* Addition of adsorption_scanner() method
* Addition of transformed fragment descriptors
* Addition of the structure rippling code

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
