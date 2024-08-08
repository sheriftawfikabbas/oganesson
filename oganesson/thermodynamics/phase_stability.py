import re
import random
import pandas as pd
import glob
import os
from ase import Atoms
from ase.io import read
import shutil
import json
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.core import Structure
import glob
import functools
import math
import ntpath

MAX_SEARCH = 10


def get_phase_decomposition(
    structure,
    phase_energy,
    vasp_phases_energies_file,
    biphasic=True,
    triphasic=True,
    tetraphasic=True,
    pentaphasic=True,
):
    energies = pd.read_csv("energies.csv", header=0, index_col="id")
    vasp_energies = pd.read_csv(vasp_phases_energies_file, header=0, index_col="id")
    fresults = open("decomposition_equations.csv", "w")
    num_Li_atoms = 0

    def get_d_fu(d):
        v = d.values()
        k = list(d.keys())
        if len(v) == 1:
            gcd = d[k[0]]
            d[k[0]] = 1
        else:
            gcd = math.gcd(int(d[k[0]]), int(d[k[1]]))
            for i in range(2, len(k)):
                gcd = math.gcd(gcd, int(d[k[i]]))
            for k in d.keys():
                d[k] /= gcd
        return d, gcd

    path = "most_stable_phases/"

    phase, gcd = get_d_fu(structure.composition.as_dict())
    phase_energy = phase_energy * len(structure) / gcd
    atoms_in_phase = list(phase.keys())
    num_atoms = sum(phase.values())
    phase["label"] = structure.formula
    # phase["Li"] += num_Li_atoms
    # phase_energy += -1.91 * num_Li_atoms

    def fix_composition(s, id):
        d = {}
        for atom_in_phase in atoms_in_phase:
            d[atom_in_phase] = 0
        for atom_in_phase in atoms_in_phase:
            if atom_in_phase in s:
                d[atom_in_phase] = s[atom_in_phase]
        label = ""
        for ss in s:
            label += ss + str(int(s[ss]))

        d["label"] = label
        d["material_id"] = id

        return d

    g = glob.glob(path + "*.json")
    phases = []
    energies = []
    for i in g:
        j = open(i)
        entry = json.load(j)
        j.close()
        file_name = ntpath.basename(i).replace(".json", "")
        id = entry["data"]["material_id"]
        structure = Structure.from_dict(entry["structure"])
        e = vasp_energies.loc[file_name]["total_energy"]
        c = entry["composition"]
        p, gcd = get_d_fu(structure.composition.as_dict())
        e /= gcd
        comp = fix_composition(c, file_name)
        phases += [comp]
        energies += [e]
        # print(comp, e, gcd)

    phases = list(filter(lambda a: a is not None, phases))

    def formula(phase, mA):
        return str(mA) + "*" + phase["label"]

    if biphasic:
        results = []
        # print('Triphasic:')
        for A in range(len(phases)):

            for mA in range(1, MAX_SEARCH):
                for mB in range(1, MAX_SEARCH):
                    phase_exists = True
                    for k in atoms_in_phase:
                        phase_exists = phase_exists and (
                            mA * phases[A][k] == mB * phase[k]
                        )
                    if phase_exists:
                        if len(results) > 0:
                            prev = results[len(results) - 1]
                            if prev[0] == A:
                                break
                        v = (
                            1000
                            / num_atoms
                            * (mA * energies[A] - mB * phase_energy)
                            / mB
                        )
                        results += [[phases[A], mA, mB, v]]
                        if v < 0:
                            line = (
                                formula(phases[A], mA)
                                + " - "
                                + formula(phase, mB)
                                + ","
                                + str(v)
                            )
                            print(line)
                            fresults.write(line + "\n")
                            fresults.flush()

        f = open(
            "phase_stability_biaphasic_" + str(num_Li_atoms) + ".json",
            "w",
        )
        json.dump(results, f)
        f.close()

    if triphasic:
        results = []
        # print('Triphasic:')
        for A in range(len(phases)):
            for B in range(len(phases)):
                if A > B:
                    for mA in range(1, MAX_SEARCH):
                        for mB in range(1, MAX_SEARCH):
                            for mC in range(1, MAX_SEARCH):
                                phase_exists = True
                                for k in atoms_in_phase:
                                    phase_exists = phase_exists and (
                                        mA * phases[A][k] + mB * phases[B][k]
                                        == mC * phase[k]
                                    )
                                if phase_exists:
                                    if len(results) > 0:
                                        prev = results[len(results) - 1]
                                        if prev[0] == A and prev[2] == B:
                                            break
                                    v = (
                                        1000
                                        / num_atoms
                                        * (
                                            mA * energies[A]
                                            + mB * energies[B]
                                            - mC * phase_energy
                                        )
                                        / mC
                                    )
                                    results += [[phases[A], mA, phases[B], mB, mC, v]]
                                    if v < 0:
                                        line = (
                                            formula(phases[A], mA)
                                            + " + "
                                            + formula(phases[B], mB)
                                            + " - "
                                            + formula(phase, mC)
                                            + ","
                                            + str(v)
                                        )
                                        print(line)
                                        fresults.write(line + "\n")
                                        fresults.flush()

        f = open(
            "phase_stability_triaphasic_" + str(num_Li_atoms) + ".json",
            "w",
        )
        json.dump(results, f)
        f.close()
    if tetraphasic:
        results = []
        # print('Tetraphasic:')
        for A in range(len(phases)):
            for B in range(len(phases)):
                for C in range(len(phases)):
                    if A > B:
                        if B > C:
                            for mA in range(1, MAX_SEARCH):
                                for mB in range(1, MAX_SEARCH):
                                    for mC in range(1, MAX_SEARCH):
                                        for mD in range(1, MAX_SEARCH):
                                            phase_exists = True
                                            for k in atoms_in_phase:
                                                phase_exists = phase_exists and (
                                                    mA * phases[A][k]
                                                    + mB * phases[B][k]
                                                    + mC * phases[C][k]
                                                    == mD * phase[k]
                                                )
                                            if phase_exists:
                                                if len(results) > 0:
                                                    prev = results[len(results) - 1]
                                                    if (
                                                        prev[0] == A
                                                        and prev[2] == B
                                                        and prev[4] == C
                                                    ):
                                                        break
                                                v = (
                                                    1000
                                                    / num_atoms
                                                    * (
                                                        mA * energies[A]
                                                        + mB * energies[B]
                                                        + mC * energies[C]
                                                        - mD * phase_energy
                                                    )
                                                    / mD
                                                )
                                                results += [
                                                    [
                                                        phases[A],
                                                        mA,
                                                        phases[B],
                                                        mB,
                                                        phases[C],
                                                        mC,
                                                        mD,
                                                        v,
                                                    ]
                                                ]

                                                if v < 0:
                                                    line = (
                                                        formula(phases[A], mA)
                                                        + " + "
                                                        + formula(phases[B], mB)
                                                        + " + "
                                                        + formula(phases[C], mC)
                                                        + " - "
                                                        + formula(phase, mD)
                                                        + ","
                                                        + str(v)
                                                    )
                                                    print(line)
                                                    fresults.write(line + "\n")
                                                    fresults.flush()

        f = open(
            "phase_stability_tetraphasic_" + str(num_Li_atoms) + ".json",
            "w",
        )
        json.dump(results, f)
        f.close()
    if pentaphasic:

        results = []
        # print('Pentaphasic:')
        for A in range(len(phases)):
            for B in range(len(phases)):
                for C in range(len(phases)):
                    for D in range(len(phases)):
                        if A > B:
                            if B > C:
                                if C > D:
                                    for mA in range(1, MAX_SEARCH):
                                        for mB in range(1, MAX_SEARCH):
                                            for mC in range(1, MAX_SEARCH):
                                                for mD in range(1, MAX_SEARCH):
                                                    for mE in range(1, MAX_SEARCH):
                                                        phase_exists = True
                                                        for k in atoms_in_phase:
                                                            phase_exists = (
                                                                phase_exists
                                                                and (
                                                                    mA * phases[A][k]
                                                                    + mB * phases[B][k]
                                                                    + mC * phases[C][k]
                                                                    + mD * phases[D][k]
                                                                    == mE * phase[k]
                                                                )
                                                            )
                                                        if phase_exists:
                                                            if len(results) > 0:
                                                                prev = results[
                                                                    len(results) - 1
                                                                ]
                                                                if (
                                                                    prev[0] == A
                                                                    and prev[2] == B
                                                                    and prev[4] == C
                                                                    and prev[6] == D
                                                                ):
                                                                    break
                                                            v = (
                                                                1000
                                                                / num_atoms
                                                                * (
                                                                    mA * energies[A]
                                                                    + mB * energies[B]
                                                                    + mC * energies[C]
                                                                    + mD * energies[D]
                                                                    - mE * phase_energy
                                                                )
                                                                / mE
                                                            )
                                                            results += [
                                                                [
                                                                    phases[A],
                                                                    mA,
                                                                    phases[B],
                                                                    mB,
                                                                    phases[C],
                                                                    mC,
                                                                    phases[D],
                                                                    mD,
                                                                    mE,
                                                                    v,
                                                                ]
                                                            ]

                                                            if v < 0:
                                                                line = (
                                                                    "("
                                                                    + formula(
                                                                        phases[A], mA
                                                                    )
                                                                    + " + "
                                                                    + formula(
                                                                        phases[B], mB
                                                                    )
                                                                    + " + "
                                                                    + formula(
                                                                        phases[C], mC
                                                                    )
                                                                    + " + "
                                                                    + formula(
                                                                        phases[D], mD
                                                                    )
                                                                    + "- "
                                                                    + formula(phase, mE)
                                                                    + ", "
                                                                    + str(v)
                                                                )
                                                                print(line)
                                                                fresults.write(
                                                                    line + "\n"
                                                                )
                                                                fresults.flush()
        f = open(
            "phase_stability_pentaphasic_" + str(num_Li_atoms) + ".json",
            "w",
        )
        json.dump(results, f)
        f.close()
    fresults.flush()
    fresults.close()
