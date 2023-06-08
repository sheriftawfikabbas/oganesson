"""
Dynamics calculations using M3GNet
"""

import contextlib
import io
import pickle
import sys
from typing import Optional, Union

import numpy as np
from ase import Atoms, units
from ase.io import Trajectory
from ase.md.nptberendsen import NPTBerendsen, Inhomogeneous_NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.andersen import Andersen
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from m3gnet.models import Potential
from m3gnet.models import M3GNet
from m3gnet.models import M3GNetCalculator


class MolecularDynamics:
    """
    Molecular dynamics class
    """

    def __init__(
        self,
        atoms: Atoms,
        potential: Union[Potential, str] = "MP-2021.2.8-EFS",
        ensemble: str = "nvt",
        thermostat: str = "anderson",
        temperature: int = 300,
        timestep: float = 1.0,
        pressure: float = 1.01325 * units.bar,
        taut: Optional[float] = None,
        taup: Optional[float] = None,
        compressibility_au: Optional[float] = None,
        trajectory: Optional[Union[str, Trajectory]] = None,
        logfile: Optional[str] = None,
        loginterval: int = 1,
        append_trajectory: bool = False,
    ):
        """

        Args:
            atoms (Atoms): atoms to run the MD
            potential (Potential): potential for calculating the energy, force,
                stress of the atoms
            ensemble (str): choose from 'nvt' or 'npt'. NPT is not tested,
                use with extra caution
            temperature (float): temperature for MD simulation, in K
            timestep (float): time step in fs
            pressure (float): pressure in eV/A^3
            taut (float): time constant for Berendsen temperature coupling
            taup (float): time constant for pressure coupling
            compressibility_au (float): compressibility of the material in A^3/eV
            trajectory (str or Trajectory): Attach trajectory object
            logfile (str): open this file for recording MD outputs
            loginterval (int): write to log file every interval steps
            append_trajectory (bool): Whether to append to prev trajectory
        """

        if isinstance(potential, str):
            potential = Potential(M3GNet.load(potential))

        if isinstance(atoms, (Structure, Molecule)):
            atoms = AseAtomsAdaptor().get_atoms(atoms)
        self.atoms = atoms
        self.atoms.set_calculator(M3GNetCalculator(potential=potential))

        if taut is None:
            taut = 100 * timestep * units.fs
        if taup is None:
            taup = 1000 * timestep * units.fs

        if ensemble.lower() == "nvt":
            if thermostat.lower() == "anderson":
                '''
                '''
                self.dyn = Andersen(andersen_prob=0.1,
                                    atoms=self.atoms,
                                    timestep=timestep * units.fs,
                                    temperature_K=temperature,
                                    trajectory=trajectory,
                                    logfile=logfile,
                                    loginterval=loginterval,
                                    append_trajectory=append_trajectory,
                                    )
            elif thermostat.lower() == "berendsen":
                self.dyn = NVTBerendsen(
                    self.atoms,
                    timestep * units.fs,
                    temperature_K=temperature,
                    taut=taut,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )

        elif ensemble.lower() == "npt":
            """

            NPT ensemble default to Inhomogeneous_NPTBerendsen thermo/barostat
            This is a more flexible scheme that fixes three angles of the unit
            cell but allows three lattice parameter to change independently.

            """

            self.dyn = Inhomogeneous_NPTBerendsen(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                pressure_au=pressure,
                taut=taut,
                taup=taup,
                compressibility_au=compressibility_au,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                # append_trajectory=append_trajectory,
                # this option is not supported in ASE at this point (I have sent merge request there)
            )

        elif ensemble.lower() == "npt_berendsen":
            """

            This is a similar scheme to the Inhomogeneous_NPTBerendsen.
            This is a less flexible scheme that fixes the shape of the
            cell - three angles are fixed and the ratios between the three
            lattice constants.

            """

            self.dyn = NPTBerendsen(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                pressure_au=pressure,
                taut=taut,
                taup=taup,
                compressibility_au=compressibility_au,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        else:
            raise ValueError("Ensemble not supported")

        self.trajectory = trajectory
        self.logfile = logfile
        self.loginterval = loginterval
        self.timestep = timestep

    def run(self, steps: int):
        """
        Thin wrapper of ase MD run
        Args:
            steps (int): number of MD steps
        Returns:

        """
        self.dyn.run(steps)

    def set_atoms(self, atoms: Atoms):
        """
        Set new atoms to run MD
        Args:
            atoms (Atoms): new atoms for running MD

        Returns:

        """
        calculator = self.atoms.calc
        self.atoms = atoms
        self.dyn.atoms = atoms
        self.dyn.atoms.set_calculator(calculator)
