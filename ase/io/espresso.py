"""Reads Quantum ESPRESSO files.

Read multiple structures and results from pw.x output files. Read
structures from pw.x input files.

Built for PWSCF v.5.3.0 but should work with earlier and later versions.
Can deal with most major functionality, with the notable exception of ibrav,
for which we only support ibrav == 0 and force CELL_PARAMETERS to be provided
explicitly.

Units are converted using CODATA 2006, as used internally by Quantum
ESPRESSO.
"""

import operator as op
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import IO, Generator

import numpy as np
from numpy.typing import NDArray

from ase.atoms import Atoms
from ase.calculators.calculator import kpts2ndarray, kpts2sizeandoffsets
from ase.calculators.singlepoint import (
    SinglePointDFTCalculator,
    SinglePointKPoint,
)
from ase.constraints import FixAtoms, FixCartesian
from ase.data import chemical_symbols
from ase.io.espresso_namelist.keys import pw_keys
from ase.io.espresso_namelist.namelist import Namelist
from ase.units import create_units
from ase.utils import deprecated, reader, writer

units = create_units('2018')

float_regex = re.compile(r'-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?')

LATE_HOP_RY = 1e-3
ACC_GATE_RY = 1e-5

ibrav_error_message = (
    'ASE does not support ibrav != 0. Note that with ibrav '
    '== 0, Quantum ESPRESSO will still detect the symmetries '
    'of your system because the CELL_PARAMETERS are defined '
    'to a high level of precision.'
)

# Section identifiers
PW_NUMBER_OF_ATOMS = r'number of atoms\/cell'
PW_ALAT = r'celldm\(1\)='
PW_FIRST_CELL = r'crystal axes:'
PW_CELL = r'CELL_PARAMETERS'
PW_FIRST_POSITIONS = r'\s+site\s+n\.\s+atom\s+.*\(alat units\)'
PW_POSITIONS = r'ATOMIC_POSITIONS'
PW_FORCES = r'Forces acting on atoms'
PW_STRESS = r'total\s+stress'
PW_MAGMOM = r'Magnetic moment per site'
PW_DIPOLE = r'Debye'
PW_DIPOLE_DIRECTION = r'Computed dipole along edir'
PW_FERMI = r'the Fermi energy is'
PW_HIGHEST_OCCUPIED = r'highest occupied level'
PW_HIGHEST_OCCUPIED_LOWEST_FREE = r'highest occupied, lowest unoccupied level'
PW_KPTS = r'number of k points\='
PW_NUMBER_OF_BANDS = r'number of Kohn-Sham states'
PW_BANDS = r'bands \(ev\):'
PW_BANDSTRUCTURE = r'End of band structure calculation'
PW_BLOCK_START = (
    r"(Program PWSCF|A final scf calculation at the relaxed structure.|coordinates at iteration)"
)
PW_RESTART = r"Atomic positions from file used, from input discarded"
PW_BLOCK_START_LSDA = r'the program is checking if it is really the minimum'
PW_FINAL_COORDS = r'Begin final coordinates'
PW_BLOCK_END = r'(number of scf cycles|Entering Dynamics:    iteration)'
PW_TOTEN = r'!\s+total energy'
PW_VERBOSITY = r"set verbosity\='high'"
PW_FORCES_SCF = r"Total SCF correction\s+="
PW_SCF_ITER_E = r'^\s+total energy\s+='
PW_SCF_ACC = r'estimated scf accuracy'
PW_SCF_CONVERGED = r'convergence has been achieved'
PW_FERMI_FAIL = r'failed to find Fermi energy'
PW_TOTAL_MAG = r'total magnetization'


def read_espresso_out(
    fileobj: IO,
    index: slice | int = -1,
    results_required: bool = True,
) -> Generator:
    """Reads Quantum ESPRESSO output files.

    The atomistic configurations as well as results, if present:

    - energy
    - forces
    - stress
    - magnetic moments
    - dipole moments
    - Fermi energy
    - k-points
    - band structure
    - cell parameters
    - positions

    Will raise a ValueError if the desired index (block) is not complete.
    Similarly, restart blocks are not supported since the only original
    atomic positions are read. This is to prevent the parser from linking
    the wrong results to the wrong configuration.

    Parameters
    ----------
    fileobj : IO
        A file like object or filename
    index : slice | int
        The index of configurations to extract.
    results_required : bool
        If True, atomistic configurations that do not have any
        associated results will not be included. This prevents double
        printed configurations and incomplete calculations from being
        returned as the final configuration with no results data.

    Yields
    ------
    structure : Atoms
        The next structure from the index slice. The Atoms has a
        SinglePointCalculator attached with any results parsed from
        the file.
    """
    output_lines = fileobj.readlines()
    len_output_lines = len(output_lines)

    indexes: dict[str, list] = {
        PW_NUMBER_OF_ATOMS: [],
        PW_ALAT: [],
        PW_FIRST_CELL: [],
        PW_CELL: [],
        PW_FIRST_POSITIONS: [],
        PW_POSITIONS: [],
        PW_FORCES: [],
        PW_STRESS: [],
        PW_MAGMOM: [],
        PW_DIPOLE: [],
        PW_DIPOLE_DIRECTION: [],
        PW_FERMI: [],
        PW_HIGHEST_OCCUPIED: [],
        PW_HIGHEST_OCCUPIED_LOWEST_FREE: [],
        PW_KPTS: [],
        PW_NUMBER_OF_BANDS: [],
        PW_BANDS: [],
        PW_BANDSTRUCTURE: [],
        PW_BLOCK_START: [],
        PW_BLOCK_END: [],
        PW_BLOCK_START_LSDA: [],
        PW_FINAL_COORDS: [],
        PW_TOTEN: [],
        PW_RESTART: [],
        PW_VERBOSITY: [],
        PW_FORCES_SCF: [],
        PW_SCF_ITER_E: [],
        PW_SCF_ACC: [],
        PW_SCF_CONVERGED: [],
        PW_FERMI_FAIL: [],
        PW_TOTAL_MAG: [],
    }

    for idx, line in enumerate(output_lines):
        for identifier in indexes:
            if re.search(identifier, line):
                indexes[identifier].append(idx)

    indexes: dict[str, NDArray] = {  # type: ignore[no-redef]
        key: np.array(value) for key, value in indexes.items()
    }

    indices_block = np.sort(
        np.hstack(
            (
                indexes[PW_BLOCK_START],
                indexes[PW_BLOCK_START_LSDA],
                indexes[PW_BLOCK_END],
            )
        )
    )
    indices_block = np.append(indices_block, len_output_lines)

    def parse_number_of_atoms(index: int) -> int:
        return int(re.findall(r'\d+', output_lines[index])[0])

    def parse_alat(index: int) -> float:
        return (
            float(re.findall(float_regex, output_lines[index])[1])
            * units['Bohr']
        )

    def parse_first_cell(index: int, alat: float) -> NDArray:
        parsed_cell = np.loadtxt(
            output_lines[index + 1 : index + 4], usecols=(3, 4, 5)
        )
        return parsed_cell * alat

    def parse_cell(index: int) -> NDArray:
        match_ = re.search(r'(bohr)|(angstrom)|(alat)', output_lines[index])
        if match_ is None:
            raise ValueError(
                'No units found in the `pw.x` output file at line {index}.'
                'Something is wrong with the output file.'
            )
        key = match_.group()
        parsed_cell = np.loadtxt(
            output_lines[index + 1 : index + 4], usecols=(0, 1, 2)
        )
        if key == 'alat':
            alat = float(re.findall(float_regex, output_lines[index])[0])
            return parsed_cell * alat * units['Bohr']
        else:
            return parsed_cell * units[key.title()]

    def parse_first_positions(
        index: int, alat: float
    ) -> tuple[NDArray, NDArray]:
        positions = (
            np.loadtxt(
                output_lines[index + 1 : index + 1 + n_atoms],
                usecols=(6, 7, 8),
                ndmin=2,
            )
            * alat
        )
        symbols = np.loadtxt(
            output_lines[index + 1 : index + 1 + n_atoms],
            usecols=1,
            converters={1: label_to_symbol},
            encoding="U",
            dtype=str,
            ndmin=1,
        )
        return positions, symbols

    def parse_positions(
        index: int, alat: float, cell: NDArray
    ) -> tuple[NDArray, NDArray]:
        match_ = re.search(
            r'(bohr)|(angstrom)|(alat)|(crystal)|(crystal_sg)',
            output_lines[index],
        )
        if match_ is None:
            raise ValueError(
                'No units found in the `pw.x` output file at line {index}.'
                'Something is wrong with the output file.'
            )
        key = match_.group()
        positions = np.loadtxt(
            output_lines[index + 1 : index + 1 + n_atoms],
            usecols=(1, 2, 3),
            ndmin=2,
        )
        if key == 'crystal_sg':
            raise ValueError('crystal_sg positions are not implemented')
        elif key == 'crystal':
            positions = np.dot(positions, cell)
        elif key == 'alat':
            positions *= alat
        else:
            positions *= units[key.title()]
        symbols = np.loadtxt(
            output_lines[index + 1 : index + 1 + n_atoms],
            usecols=0,
            converters=label_to_symbol,
            dtype=str,
            encoding="U",
            ndmin=1,
        )
        return positions, symbols

    def parse_energy(index: int) -> float:
        return (
            float(re.findall(float_regex, output_lines[index])[0]) * units['Ry']
        )

    def parse_forces(index: int) -> NDArray:
        offset = 4 if 'negative rho' in output_lines[index + 2] else 2
        forces = np.loadtxt(
            output_lines[index + offset : index + offset + n_atoms],
            usecols=(6, 7, 8),
        )
        return forces * units['Ry'] / units['Bohr']

    def parse_stress(index: int) -> NDArray:
        stress = np.loadtxt(
            output_lines[index + 1 : index + 4], usecols=(0, 1, 2)
        )
        return stress * units['Ry'] / (units['Bohr'] ** 3)

    def parse_magmom(index: int) -> NDArray:
        magmoms = np.loadtxt(
            output_lines[index + 1 : index + 1 + n_atoms], usecols=-1
        )
        return magmoms

    def parse_dipole(index: int) -> float:
        return (
            float(re.findall(float_regex, output_lines[index])[-1])
            * units['Debye']
        )

    def parse_fermi(index: int) -> float:
        return float(re.findall(float_regex, output_lines[index])[0])

    def parse_kpts_and_weights(index: int) -> tuple[NDArray, NDArray]:
        len_kpts = int(re.findall(float_regex, output_lines[index])[0])

        kpts_and_weights = np.loadtxt(
            output_lines[index + 2 : index + 2 + len_kpts],
            converters=lambda x: float(x[:-2].strip(')')),
            usecols=(4, 5, 6, -1),
            encoding="U",
            ndmin=2,
        )

        kpts_coord = kpts_and_weights[:, :3]
        kpts_weights = kpts_and_weights[:, -1]

        # QE prints the k-points in units of 2*pi/alat, we return it in
        # cartesian by multiplying with the inverse of the reciprocal cell
        return np.dot(kpts_coord / alat, cell.T), kpts_weights

    def parse_number_of_bands(index: int) -> int:
        return int(re.findall(r'\d+', output_lines[index])[0])

    def parse_dipole_direction(index: int) -> int:
        return int(re.findall(r'\d+', output_lines[index])[0]) - 1

    def parse_bands(index: int) -> NDArray:
        offset = 1 if n_bands % 8 == 0 else 2
        bands = [
            re.findall(float_regex, line)
            for line in output_lines[
                index + 1 : index + 1 + n_bands // 8 + offset
            ]
        ]

        return np.hstack(bands).astype(float)

    def parse_scf_force_correction(index: int) -> float:
        return float(re.findall(float_regex, output_lines[index])[-1]) * units[
            'Ry'
        ] / units['Bohr']

    def parse_first_float(index: int) -> float:
        return float(re.findall(float_regex, output_lines[index])[0])

    properties = list(indexes.keys())

    base_properties = [
        PW_NUMBER_OF_ATOMS,
        PW_ALAT,
    ]
    results_properties = (
        [
            PW_TOTEN,
            PW_FORCES,
        ]
        if results_required
        else []
    )
    required_properties = base_properties + results_properties

    property_only_at_start = [
        PW_NUMBER_OF_ATOMS,
        PW_NUMBER_OF_BANDS,
        PW_ALAT,
        PW_CELL,
        PW_KPTS,
    ]

    for num_block, (past, future) in enumerate(
        zip(indices_block[:-1][index], indices_block[1:][index])
    ):
        current_indices: dict[str, NDArray] = {}

        for property_ in properties:
            is_property = np.logical_and(
                past <= indexes[property_], indexes[property_] < future
            )
            current_indices[property_] = indexes[property_][is_property]

        if current_indices[PW_RESTART].size:
            raise ValueError(
                'The output file contains a restart block, which is not'
                'supported by this parser.'
            )

        if not (
            current_indices[PW_POSITIONS].size
            or current_indices[PW_FIRST_POSITIONS].size
        ):
            raise ValueError(
                'No positions found in the `pw.x` output file at requested'
                f'block {num_block} starting at line {past}.'
            )

        # If alat is None, we are not at the start of a calculation and we need
        # to find the alat printed at the last start
        def seek_last_property(property_):
            # We look for past starts (Program PWSCF or A final scf calculation)
            past_starts = indexes[PW_BLOCK_START][
                indexes[PW_BLOCK_START] < future
            ]

            # We look for the property printed after the last start
            past_property = indexes[property_][
                np.logical_and(
                    indexes[property_] < future,
                    np.abs(future - indexes[property_])
                    < np.abs(future - past_starts[-1]),
                )
            ]

            return past_property

        for property_ in property_only_at_start:
            if current_indices[property_].size == 0:
                # PW_CELL is a special case, it can be the first cell or the
                # cell printed after the positions in case of vc-relax
                if property_ == PW_CELL:
                    current_indices[PW_FIRST_CELL] = seek_last_property(
                        PW_FIRST_CELL
                    )
                else:
                    current_indices[property_] = seek_last_property(property_)

        n_atoms = parse_number_of_atoms(current_indices[PW_NUMBER_OF_ATOMS][-1])
        n_bands = parse_number_of_bands(current_indices[PW_NUMBER_OF_BANDS][-1])

        if current_indices[PW_ALAT].size:
            alat = parse_alat(current_indices[PW_ALAT][-1])
        else:
            raise ValueError(
                'No alat found in the `pw.x` output file at block'
                f'{num_block} starting at line {past}.'
                'Something is wrong with the output file.'
            )

        if not (
            current_indices[PW_FIRST_CELL].size or current_indices[PW_CELL].size
        ):
            raise ValueError(
                'No cell found in the `pw.x` output file at '
                f'requested block {num_block} starting at line {past}.'
            )

        if current_indices[PW_FIRST_CELL].size:
            cell = parse_first_cell(current_indices[PW_FIRST_CELL][-1], alat)
            current_indices[PW_CELL] = current_indices[PW_FIRST_CELL]
        else:
            # If the final coordinates are present, two cells can be within the
            # same block, we need to check if that's the case.
            if current_indices[PW_CELL].size > 1:
                if current_indices[PW_FINAL_COORDS].size == 0:
                    raise ValueError(
                        'Multiple cell blocks found in the `pw.x` output file'
                        f'at block {num_block} starting at line {past}.'
                    )
            cell = parse_cell(current_indices[PW_CELL][-1])

        if current_indices[PW_FIRST_POSITIONS].size:
            positions, symbols = parse_first_positions(
                current_indices[PW_FIRST_POSITIONS][-1],
                alat,
            )
            current_indices[PW_POSITIONS] = current_indices[PW_FIRST_POSITIONS]
        else:
            if current_indices[PW_POSITIONS].size > 1:
                if current_indices[PW_FINAL_COORDS].size == 0:
                    raise ValueError(
                        'Multiple positions blocks found in the `pw.x` output'
                        f' file at block {num_block} starting at line {past}.'
                    )
            positions, symbols = parse_positions(
                current_indices[PW_POSITIONS][-1], alat, cell
            )

        for property_ in required_properties:
            if current_indices[property_].size == 0:
                raise ValueError(
                    'Required properties are missing from the output file:'
                    f'{property_} at requested block {num_block}, starting at'
                    f' line {past}.'
                )

        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=cell,
            pbc=True,
        )

        if current_indices[PW_HIGHEST_OCCUPIED].size:
            current_indices[PW_FERMI] = current_indices[PW_HIGHEST_OCCUPIED]
        elif current_indices[PW_HIGHEST_OCCUPIED_LOWEST_FREE].size:
            current_indices[PW_FERMI] = current_indices[
                PW_HIGHEST_OCCUPIED_LOWEST_FREE
            ]

        if current_indices[PW_FERMI].size:
            fermi = parse_fermi(current_indices[PW_FERMI][-1])
        else:
            fermi = None

        kpts = []
        if current_indices[PW_VERBOSITY].size == 0:
            ibzkpts, weights = parse_kpts_and_weights(
                current_indices[PW_KPTS][-1]
            )

            for kpt_index, idx in enumerate(current_indices.get(PW_BANDS, [])):
                spin = kpt_index // len(ibzkpts)
                ibzkpts_index = kpt_index % len(ibzkpts)
                eigenvalues = parse_bands(idx)
                kpts.append(
                    SinglePointKPoint(
                        weights[ibzkpts_index],
                        spin,
                        ibzkpts[ibzkpts_index],
                        eps_n=eigenvalues,
                    )
                )
        else:
            ibzkpts, weights = None, None

        if current_indices[PW_DIPOLE].size:
            dipole_direction = parse_dipole_direction(
                current_indices[PW_DIPOLE_DIRECTION][-1]
            )
            dipole = (
                parse_dipole(current_indices[PW_DIPOLE][-1])
                * np.eye(3)[dipole_direction, dipole_direction]
            )
        else:
            dipole = None

        properties_left_to_parse = {
            PW_TOTEN: parse_energy,
            PW_FORCES: parse_forces,
            PW_STRESS: parse_stress,
            PW_MAGMOM: parse_magmom,
            PW_FORCES_SCF: parse_scf_force_correction,
        }

        computed_properties = {}

        for property_ in properties_left_to_parse:
            if current_indices[property_].size:
                computed_properties[property_] = properties_left_to_parse[
                    property_
                ](current_indices[property_][-1])

        forces = computed_properties.get(PW_FORCES)
        forces_correction = computed_properties.get(PW_FORCES_SCF)

        if forces is not None and forces_correction is not None:
            total_force = np.linalg.norm(forces)

            if total_force > 0.1 and forces_correction / total_force > 0.05:
                raise ValueError(
                    'SCF correction is too large compared to the forces.'
                )
            atoms.info['qe_scf_force_correction'] = forces_correction

        # ---- per-frame SCF health checks (pw.x text diagnostics) ----
        if current_indices[PW_FERMI_FAIL].size:
            raise ValueError(
                'The SCF emitted "failed to find Fermi energy" for this'
                f' configuration (block {num_block}).'
            )

        scf_converged = bool(current_indices[PW_SCF_CONVERGED].size)
        atoms.info['qe_scf_converged'] = scf_converged
        atoms.info['qe_scf_iterations'] = int(
            current_indices[PW_SCF_ITER_E].size
        )
        if current_indices[PW_TOTAL_MAG].size:
            atoms.info['qe_total_magnetization'] = parse_first_float(
                current_indices[PW_TOTAL_MAG][-1]
            )

        if scf_converged:
            iteration_energies = [
                parse_first_float(i)
                for i in current_indices[PW_SCF_ITER_E]
            ]
            accuracies = [
                parse_first_float(i) for i in current_indices[PW_SCF_ACC]
            ]
            for i in range(1, len(iteration_energies)):
                if i - 1 >= len(accuracies):
                    break
                jump = abs(
                    iteration_energies[i] - iteration_energies[i - 1]
                )
                if accuracies[i - 1] < ACC_GATE_RY and jump > LATE_HOP_RY:
                    raise ValueError(
                        f'Late SCF energy hop of {jump:.1e} Ry after the'
                        f' cycle reported accuracy {accuracies[i - 1]:.1e}'
                        f' Ry, yet claimed convergence (block {num_block}).'
                    )

        calc = SinglePointDFTCalculator(
            atoms,
            energy=computed_properties.get(PW_TOTEN),
            free_energy=computed_properties.get(PW_TOTEN),
            forces=computed_properties.get(PW_FORCES),
            stress=computed_properties.get(PW_STRESS),
            magmoms=computed_properties.get(PW_MAGMOM),
            efermi=fermi,
            kpts=kpts,
            ibzkpts=ibzkpts,
            dipole=dipole,
        )
        atoms.calc = calc

        yield atoms


@reader
def read_espresso_in(fileobj):
    """Parse a Quantum ESPRESSO input files, '.in', '.pwi'.

    ESPRESSO inputs are generally a fortran-namelist format with custom
    blocks of data. The namelist is parsed as a dict and an atoms object
    is constructed from the included information.

    Parameters
    ----------
    fileobj : file | str
        A file-like object that supports line iteration with the contents
        of the input file, or a filename.

    Returns
    -------
    atoms : Atoms
        Structure defined in the input file.

    Raises
    ------
    KeyError
        Raised for missing keys that are required to process the file
    """
    # parse namelist section and extract remaining lines
    data, card_lines = read_fortran_namelist(fileobj)

    # get the cell if ibrav=0
    if 'system' not in data:
        raise KeyError('Required section &SYSTEM not found.')
    elif 'ibrav' not in data['system']:
        raise KeyError('ibrav is required in &SYSTEM')
    elif data['system']['ibrav'] == 0:
        # celldm(1) is in Bohr, A is in angstrom. celldm(1) will be
        # used even if A is also specified.
        if 'celldm(1)' in data['system']:
            alat = data['system']['celldm(1)'] * units['Bohr']
        elif 'A' in data['system']:
            alat = data['system']['A']
        else:
            alat = None
        cell, _ = get_cell_parameters(card_lines, alat=alat)
    else:
        raise ValueError(ibrav_error_message)

    # species_info holds some info for each element
    species_card = get_atomic_species(
        card_lines, n_species=data['system']['ntyp']
    )
    species_info = {}
    for ispec, (label, weight, pseudo) in enumerate(species_card):
        symbol = label_to_symbol(label)

        # starting_magnetization is in fractions of valence electrons
        magnet_key = f'starting_magnetization({ispec + 1})'
        magmom = data['system'].get(magnet_key, 0.0)
        species_info[symbol] = {
            'weight': weight,
            'pseudo': pseudo,
            'magmom': magmom,
        }

    positions_card = get_atomic_positions(
        card_lines, n_atoms=data['system']['nat'], cell=cell, alat=alat
    )

    symbols = [label_to_symbol(position[0]) for position in positions_card]
    positions = [position[1] for position in positions_card]
    constraint_flags = [position[2] for position in positions_card]
    magmoms = [species_info[symbol]['magmom'] for symbol in symbols]

    # TODO: put more info into the atoms object
    # e.g magmom, forces.
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=cell,
        pbc=True,
        magmoms=magmoms,
    )
    atoms.set_constraint(convert_constraint_flags(constraint_flags))

    return atoms


def get_atomic_positions(lines, n_atoms, cell=None, alat=None):
    """Parse atom positions from ATOMIC_POSITIONS card.

    Parameters
    ----------
    lines : list[str]
        A list of lines containing the ATOMIC_POSITIONS card.
    n_atoms : int
        Expected number of atoms. Only this many lines will be parsed.
    cell : np.array
        Unit cell of the crystal. Only used with crystal coordinates.
    alat : float
        Lattice parameter for atomic coordinates. Only used for alat case.

    Returns
    -------
    positions : list[(str, (float, float, float), (int, int, int))]
        A list of the ordered atomic positions in the format:
        label, (x, y, z), (if_x, if_y, if_z)
        Force multipliers are set to None if not present.

    Raises
    ------
    ValueError
        Any problems parsing the data result in ValueError

    """

    positions = None
    # no blanks or comment lines, can the consume n_atoms lines for positions
    trimmed_lines = (line for line in lines if line.strip() and line[0] != '#')

    for line in trimmed_lines:
        if line.strip().startswith('ATOMIC_POSITIONS'):
            if positions is not None:
                raise ValueError('Multiple ATOMIC_POSITIONS specified')
            # Priority and behaviour tested with QE 5.3
            if 'crystal_sg' in line.lower():
                raise NotImplementedError('CRYSTAL_SG not implemented')
            elif 'crystal' in line.lower():
                cell = cell
            elif 'bohr' in line.lower():
                cell = np.identity(3) * units['Bohr']
            elif 'angstrom' in line.lower():
                cell = np.identity(3)
            # elif 'alat' in line.lower():
            #     cell = np.identity(3) * alat
            else:
                if alat is None:
                    raise ValueError(
                        'Set lattice parameter in &SYSTEM for alat coordinates'
                    )
                # Always the default, will be DEPRECATED as mandatory
                # in future
                cell = np.identity(3) * alat

            positions = []
            for _ in range(n_atoms):
                split_line = next(trimmed_lines).split()
                # These can be fractions and other expressions
                position = np.dot(
                    (
                        infix_float(split_line[1]),
                        infix_float(split_line[2]),
                        infix_float(split_line[3]),
                    ),
                    cell,
                )
                if len(split_line) > 4:
                    force_mult = tuple(int(split_line[i]) for i in (4, 5, 6))
                else:
                    force_mult = None

                positions.append((split_line[0], position, force_mult))

    return positions


def get_atomic_species(lines, n_species):
    """Parse atomic species from ATOMIC_SPECIES card.

    Parameters
    ----------
    lines : list[str]
        A list of lines containing the ATOMIC_POSITIONS card.
    n_species : int
        Expected number of atom types. Only this many lines will be parsed.

    Returns
    -------
    species : list[(str, float, str)]

    Raises
    ------
    ValueError
        Any problems parsing the data result in ValueError
    """

    species = None
    # no blanks or comment lines, can the consume n_atoms lines for positions
    trimmed_lines = (
        line.strip()
        for line in lines
        if line.strip() and not line.startswith('#')
    )

    for line in trimmed_lines:
        if line.startswith('ATOMIC_SPECIES'):
            if species is not None:
                raise ValueError('Multiple ATOMIC_SPECIES specified')

            species = []
            for _dummy in range(n_species):
                label_weight_pseudo = next(trimmed_lines).split()
                species.append(
                    (
                        label_weight_pseudo[0],
                        float(label_weight_pseudo[1]),
                        label_weight_pseudo[2],
                    )
                )

    return species


def get_cell_parameters(lines, alat=None):
    """Parse unit cell from CELL_PARAMETERS card.

    Parameters
    ----------
    lines : list[str]
        A list with lines containing the CELL_PARAMETERS card.
    alat : float | None
        Unit of lattice vectors in Angstrom. Only used if the card is
        given in units of alat. alat must be None if CELL_PARAMETERS card
        is in Bohr or Angstrom. For output files, alat will be parsed from
        the card header and used in preference to this value.

    Returns
    -------
    cell : np.array | None
        Cell parameters as a 3x3 array in Angstrom. If no cell is found
        None will be returned instead.
    cell_alat : float | None
        If a value for alat is given in the card header, this is also
        returned, otherwise this will be None.

    Raises
    ------
    ValueError
        If CELL_PARAMETERS are given in units of bohr or angstrom
        and alat is not
    """

    cell = None
    cell_alat = None
    # no blanks or comment lines, can take three lines for cell
    trimmed_lines = (line for line in lines if line.strip() and line[0] != '#')

    for line in trimmed_lines:
        if line.strip().startswith('CELL_PARAMETERS'):
            if cell is not None:
                # multiple definitions
                raise ValueError('CELL_PARAMETERS specified multiple times')
            # Priority and behaviour tested with QE 5.3
            if 'bohr' in line.lower():
                if alat is not None:
                    raise ValueError(
                        'Lattice parameters given in '
                        '&SYSTEM celldm/A and CELL_PARAMETERS '
                        'bohr'
                    )
                cell_units = units['Bohr']
            elif 'angstrom' in line.lower():
                if alat is not None:
                    raise ValueError(
                        'Lattice parameters given in '
                        '&SYSTEM celldm/A and CELL_PARAMETERS '
                        'angstrom'
                    )
                cell_units = 1.0
            elif 'alat' in line.lower():
                # Output file has (alat = value) (in Bohrs)
                if '=' in line:
                    alat = float(line.strip(') \n').split()[-1]) * units['Bohr']
                    cell_alat = alat
                elif alat is None:
                    raise ValueError(
                        'Lattice parameters must be set in '
                        '&SYSTEM for alat units'
                    )
                cell_units = alat
            elif alat is None:
                # may be DEPRECATED in future
                cell_units = units['Bohr']
            else:
                # may be DEPRECATED in future
                cell_units = alat
            # Grab the parameters; blank lines have been removed
            cell = [
                [ffloat(x) for x in next(trimmed_lines).split()[:3]],
                [ffloat(x) for x in next(trimmed_lines).split()[:3]],
                [ffloat(x) for x in next(trimmed_lines).split()[:3]],
            ]
            cell = np.array(cell) * cell_units

    return cell, cell_alat


def convert_constraint_flags(constraint_flags):
    """Convert Quantum ESPRESSO constraint flags to ASE Constraint objects.

    Parameters
    ----------
    constraint_flags : list[tuple[int, int, int]]
        List of constraint flags (0: fixed, 1: moved) for all the atoms.
        If the flag is None, there are no constraints on the atom.

    Returns
    -------
    constraints : list[FixAtoms | FixCartesian]
        List of ASE Constraint objects.
    """
    constraints = []
    for i, constraint in enumerate(constraint_flags):
        if constraint is None:
            continue
        # mask: False (0): moved, True (1): fixed
        mask = ~np.asarray(constraint, bool)
        constraints.append(FixCartesian(i, mask))
    return canonicalize_constraints(constraints)


def canonicalize_constraints(constraints):
    """Canonicalize ASE FixCartesian constraints.

    If the given FixCartesian constraints share the same `mask`, they can be
    merged into one. Further, if `mask == (True, True, True)`, they can be
    converted as `FixAtoms`. This method "canonicalizes" FixCartesian objects
    in such a way.

    Parameters
    ----------
    constraints : List[FixCartesian]
        List of ASE FixCartesian constraints.

    Returns
    -------
    constrants_canonicalized : List[FixAtoms | FixCartesian]
        List of ASE Constraint objects.
    """
    # https://docs.python.org/3/library/collections.html#defaultdict-examples
    indices_for_masks = defaultdict(list)
    for constraint in constraints:
        key = tuple((constraint.mask).tolist())
        indices_for_masks[key].extend(constraint.index.tolist())

    constraints_canonicalized = []
    for mask, indices in indices_for_masks.items():
        if mask == (False, False, False):  # no directions are fixed
            continue
        if mask == (True, True, True):  # all three directions are fixed
            constraints_canonicalized.append(FixAtoms(indices))
        else:
            constraints_canonicalized.append(FixCartesian(indices, mask))

    return constraints_canonicalized


def str_to_value(string):
    """Attempt to convert string into int, float (including fortran double),
    or bool, in that order, otherwise return the string.
    Valid (case-insensitive) bool values are: '.true.', '.t.', 'true'
    and 't' (or false equivalents).

    Parameters
    ----------
    string : str
        Test to parse for a datatype

    Returns
    -------
    value : any
        Parsed string as the most appropriate datatype of int, float,
        bool or string.
    """

    # Just an integer
    try:
        return int(string)
    except ValueError:
        pass
    # Standard float
    try:
        return float(string)
    except ValueError:
        pass
    # Fortran double
    try:
        return ffloat(string)
    except ValueError:
        pass

    # possible bool, else just the raw string
    if string.lower() in ('.true.', '.t.', 'true', 't'):
        return True
    elif string.lower() in ('.false.', '.f.', 'false', 'f'):
        return False
    else:
        return string.strip("'")


def read_fortran_namelist(fileobj):
    """Takes a fortran-namelist formatted file and returns nested
    dictionaries of sections and key-value data, followed by a list
    of lines of text that do not fit the specifications.
    Behaviour is taken from Quantum ESPRESSO 5.3. Parses fairly
    convoluted files the same way that QE should, but may not get
    all the MANDATORY rules and edge cases for very non-standard files
    Ignores anything after '!' in a namelist, split pairs on ','
    to include multiple key=values on a line, read values on section
    start and end lines, section terminating character, '/', can appear
    anywhere on a line. All of these are ignored if the value is in 'quotes'.

    Parameters
    ----------
    fileobj : file
        An open file-like object.

    Returns
    -------
    data : dict[str, dict]
        Dictionary for each section in the namelist with
        key = value pairs of data.
    additional_cards : list[str]
        Any lines not used to create the data,
        assumed to belong to 'cards' in the input file.
    """

    data = {}
    card_lines = []
    in_namelist = False
    section = 'none'  # can't be in a section without changing this

    for line in fileobj:
        # leading and trailing whitespace never needed
        line = line.strip()
        if line.startswith('&'):
            # inside a namelist
            section = line.split()[0][1:].lower()  # case insensitive
            if section in data:
                # Repeated sections are completely ignored.
                # (Note that repeated keys overwrite within a section)
                section = '_ignored'
            data[section] = {}
            in_namelist = True
        if not in_namelist and line:
            # Stripped line is Truthy, so safe to index first character
            if line[0] not in ('!', '#'):
                card_lines.append(line)
        if in_namelist:
            # parse k, v from line:
            key = []
            value = None
            in_quotes = False
            for character in line:
                if character == ',' and value is not None and not in_quotes:
                    # finished value:
                    data[section][''.join(key).strip()] = str_to_value(
                        ''.join(value).strip()
                    )
                    key = []
                    value = None
                elif character == '=' and value is None and not in_quotes:
                    # start writing value
                    value = []
                elif character == "'":
                    # only found in value anyway
                    in_quotes = not in_quotes
                    value.append("'")
                elif character == '!' and not in_quotes:
                    break
                elif character == '/' and not in_quotes:
                    in_namelist = False
                    break
                elif value is not None:
                    value.append(character)
                else:
                    key.append(character)
            if value is not None:
                data[section][''.join(key).strip()] = str_to_value(
                    ''.join(value).strip()
                )

    return Namelist(data), card_lines


def ffloat(string):
    """Parse float from fortran compatible float definitions.

    In fortran exponents can be defined with 'd' or 'q' to symbolise
    double or quad precision numbers. Double precision numbers are
    converted to python floats and quad precision values are interpreted
    as numpy longdouble values (platform specific precision).

    Parameters
    ----------
    string : str
        A string containing a number in fortran real format

    Returns
    -------
    value : float | np.longdouble
        Parsed value of the string.

    Raises
    ------
    ValueError
        Unable to parse a float value.

    """

    if 'q' in string.lower():
        return np.longdouble(string.lower().replace('q', 'e'))
    else:
        return float(string.lower().replace('d', 'e'))


def label_to_symbol(label: str) -> str:
    """Convert a valid espresso ATOMIC_SPECIES label to a chemical symbol.

    Parameters
    ----------
    label : str
        Chemical symbol X (1 or 2 characters, case-insensitive)
        or chemical symbol plus a number or a letter, as in
        "Xn" (e.g. Fe1) or "X_*" or "X-*" (e.g. C1, C_h).
        Max total length cannot exceed 3 characters.

    Returns
    -------
    str
        The best matching chemical symbol from ase.utils.chemical_symbols

    Raises
    ------
    ValueError
        If label is empty, too long (>3 chars), or no matching symbol found

    Examples
    --------
    >>> label_to_symbol('Fe')
    'Fe'
    >>> label_to_symbol('fe1')
    'Fe'
    >>> label_to_symbol('C_h')
    'C'
    """
    if not label:
        raise ValueError('Label cannot be empty')

    if len(label) > 3:
        raise ValueError(
            f"Label '{label}' exceeds maximum length of 3 characters"
        )

    # Extract potential chemical symbol using regex
    # Matches: Single char, or two chars, ignoring any trailing numbers/characters
    match = re.match(r'^([A-Za-z]{1,2})', label)
    if not match:
        raise ValueError(
            f"Label '{label}' does not start with valid chemical symbol characters"
        )

    potential_symbol = match.group(1)

    # Try two-character symbol first (if available)
    if len(potential_symbol) == 2:
        symbol = potential_symbol[0].upper() + potential_symbol[1].lower()
        if symbol in chemical_symbols:
            return symbol

    # Try single-character symbol
    symbol = potential_symbol[0].upper()
    if symbol in chemical_symbols:
        return symbol

    raise ValueError(
        f"Could not find matching chemical symbol for label '{label}'"
    )


def infix_float(text):
    """Parse simple infix maths into a float for compatibility with
    Quantum ESPRESSO ATOMIC_POSITIONS cards. Note: this works with the
    example, and most simple expressions, but the capabilities of
    the two parsers are not identical. Will also parse a normal float
    value properly, but slowly.

    >>> infix_float('1/2*3^(-1/2)')
    0.28867513459481287

    Parameters
    ----------
    text : str
        An arithmetic expression using +, -, *, / and ^, including brackets.

    Returns
    -------
    value : float
        Result of the mathematical expression.

    """

    def middle_brackets(full_text):
        """Extract text from innermost brackets."""
        start, end = 0, len(full_text)
        for idx, char in enumerate(full_text):
            if char == '(':
                start = idx
            if char == ')':
                end = idx + 1
                break
        return full_text[start:end]

    def eval_no_bracket_expr(full_text):
        """Calculate value of a mathematical expression, no brackets."""
        exprs = [('+', op.add), ('*', op.mul), ('/', op.truediv), ('^', op.pow)]
        full_text = full_text.lstrip('(').rstrip(')')
        try:
            return float(full_text)
        except ValueError:
            for symbol, func in exprs:
                if symbol in full_text:
                    left, right = full_text.split(symbol, 1)  # single split
                    return func(
                        eval_no_bracket_expr(left), eval_no_bracket_expr(right)
                    )

    while '(' in text:
        middle = middle_brackets(text)
        text = text.replace(middle, f'{eval_no_bracket_expr(middle)}')

    return float(eval_no_bracket_expr(text))


# Number of valence electrons in the pseudopotentials recommended by
# http://materialscloud.org/sssp/. These are just used as a fallback for
# calculating initial magetization values which are given as a fraction
# of valence electrons.
SSSP_VALENCE = [
    0,
    1.0,
    2.0,
    3.0,
    4.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
    15.0,
    16.0,
    17.0,
    18.0,
    19.0,
    20.0,
    13.0,
    14.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
    15.0,
    16.0,
    17.0,
    18.0,
    19.0,
    12.0,
    13.0,
    14.0,
    15.0,
    6.0,
    7.0,
    18.0,
    9.0,
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
    15.0,
    16.0,
    17.0,
    18.0,
    19.0,
    20.0,
    21.0,
    22.0,
    23.0,
    24.0,
    25.0,
    36.0,
    27.0,
    14.0,
    15.0,
    30.0,
    15.0,
    32.0,
    19.0,
    12.0,
    13.0,
    14.0,
    15.0,
    16.0,
    18.0,
]


def kspacing_to_grid(atoms, spacing, calculated_spacing=None):
    """
    Calculate the kpoint mesh that is equivalent to the given spacing
    in reciprocal space (units Angstrom^-1). The number of kpoints is each
    dimension is rounded up (compatible with CASTEP).

    Parameters
    ----------
    atoms: ase.Atoms
        A structure that can have get_reciprocal_cell called on it.
    spacing: float
        Minimum K-Point spacing in $A^{-1}$.
    calculated_spacing : list
        If a three item list (or similar mutable sequence) is given the
        members will be replaced with the actual calculated spacing in
        $A^{-1}$.

    Returns
    -------
    kpoint_grid : [int, int, int]
        MP grid specification to give the required spacing.

    """
    # No factor of 2pi in ase, everything in A^-1
    # reciprocal dimensions
    r_x, r_y, r_z = np.linalg.norm(atoms.cell.reciprocal(), axis=1)

    kpoint_grid = [
        int(r_x / spacing) + 1,
        int(r_y / spacing) + 1,
        int(r_z / spacing) + 1,
    ]

    for i, _ in enumerate(kpoint_grid):
        if not atoms.pbc[i]:
            kpoint_grid[i] = 1

    if calculated_spacing is not None:
        calculated_spacing[:] = [
            r_x / kpoint_grid[0],
            r_y / kpoint_grid[1],
            r_z / kpoint_grid[2],
        ]

    return kpoint_grid


def format_atom_position(atom, crystal_coordinates, mask='', tidx=None):
    """Format one line of atomic positions in
    Quantum ESPRESSO ATOMIC_POSITIONS card.

    >>> for atom in make_supercell(bulk('Li', 'bcc'), np.ones(3)-np.eye(3)):
    >>>     format_atom_position(atom, True)
    Li 0.0000000000 0.0000000000 0.0000000000
    Li 0.5000000000 0.5000000000 0.5000000000

    Parameters
    ----------
    atom : Atom
        A structure that has symbol and [position | (a, b, c)].
    crystal_coordinates: bool
        Whether the atomic positions should be written to the QE input file in
        absolute (False, default) or relative (crystal) coordinates (True).
    mask, optional : str
        String of ndim=3 0 or 1 for constraining atomic positions.
    tidx, optional : int
        Magnetic type index.

    Returns
    -------
    atom_line : str
        Input line for atom position
    """
    if crystal_coordinates:
        coords = [atom.a, atom.b, atom.c]
    else:
        coords = atom.position
    line_fmt = '{atom.symbol}'
    inps = dict(atom=atom)
    if tidx is not None:
        line_fmt += '{tidx}'
        inps['tidx'] = tidx
    line_fmt += ' {coords[0]:.10f} {coords[1]:.10f} {coords[2]:.10f} '
    inps['coords'] = coords
    line_fmt += ' ' + mask + '\n'
    astr = line_fmt.format(**inps)
    return astr


@writer
def write_espresso_in(
    fd,
    atoms,
    input_data=None,
    pseudopotentials=None,
    kspacing=None,
    kpts=None,
    koffset=(0, 0, 0),
    crystal_coordinates=False,
    additional_cards=None,
    **kwargs,
):
    """
    Create an input file for pw.x.

    Use set_initial_magnetic_moments to turn on spin, if nspin is set to 2
    with no magnetic moments, they will all be set to 0.0. Magnetic moments
    will be converted to the QE units (fraction of valence electrons) using
    any pseudopotential files found, or a best guess for the number of
    valence electrons.

    Units are not converted for any other input data, so use Quantum ESPRESSO
    units (Usually Ry or atomic units).

    Keys with a dimension (e.g. Hubbard_U(1)) will be incorporated as-is
    so the `i` should be made to match the output.

    Implemented features:

    - Conversion of :class:`ase.constraints.FixAtoms` and
      :class:`ase.constraints.FixCartesian`.
    - ``starting_magnetization`` derived from the ``magmoms`` and
      pseudopotentials (searches default paths for pseudo files.)
    - Automatic assignment of options to their correct sections.

    Not implemented:

    - Non-zero values of ibrav
    - Lists of k-points
    - Other constraints
    - Hubbard parameters
    - Validation of the argument types for input
    - Validation of required options

    Parameters
    ----------
    fd: file | str
        A file to which the input is written.
    atoms: Atoms
        A single atomistic configuration to write to ``fd``.
    input_data: dict
        A flat or nested dictionary with input parameters for pw.x
    pseudopotentials: dict
        A filename for each atomic species, e.g.
        {'O': 'O.pbe-rrkjus.UPF', 'H': 'H.pbe-rrkjus.UPF'}.
        A dummy name will be used if none are given.
    kspacing: float
        Generate a grid of k-points with this as the minimum distance,
        in A^-1 between them in reciprocal space. If set to None, kpts
        will be used instead.
    kpts: (int, int, int), dict or np.ndarray
        If ``kpts`` is a tuple (or list) of 3 integers, it is interpreted
        as the dimensions of a Monkhorst-Pack grid.
        If ``kpts`` is set to ``None``, only the Γ-point will be included
        and QE will use routines optimized for Γ-point-only calculations.
        Compared to Γ-point-only calculations without this optimization
        (i.e. with ``kpts=(1, 1, 1)``), the memory and CPU requirements
        are typically reduced by half.
        If kpts is a dict, it will either be interpreted as a path
        in the Brillouin zone (*) if it contains the 'path' keyword,
        otherwise it is converted to a Monkhorst-Pack grid (**).
        If ``kpts`` is a NumPy array, the raw k-points will be passed to
        Quantum Espresso as given in the array (in crystal coordinates).
        Must be of shape (n_kpts, 4). The fourth column contains the
        k-point weights.
        (*) see ase.dft.kpoints.bandpath
        (**) see ase.calculators.calculator.kpts2sizeandoffsets
    koffset: (int, int, int)
        Offset of kpoints in each direction. Must be 0 (no offset) or
        1 (half grid offset). Setting to True is equivalent to (1, 1, 1).
    crystal_coordinates: bool
        Whether the atomic positions should be written to the QE input file in
        absolute (False, default) or relative (crystal) coordinates (True).

    """

    # Convert to a namelist to make working with parameters much easier
    # Note that the name ``input_data`` is chosen to prevent clash with
    # ``parameters`` in Calculator objects
    input_parameters = Namelist(input_data)
    input_parameters.to_nested('pw', **kwargs)

    # Convert ase constraints to QE constraints
    # Nx3 array of force multipliers matches what QE uses
    # Do this early so it is available when constructing the atoms card
    moved = np.ones((len(atoms), 3), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            moved[constraint.index] = False
        elif isinstance(constraint, FixCartesian):
            moved[constraint.index] = ~constraint.mask
        else:
            warnings.warn(f'Ignored unknown constraint {constraint}')
    masks = []
    for atom in atoms:
        # only inclued mask if something is fixed
        if not all(moved[atom.index]):
            mask = ' {:d} {:d} {:d}'.format(*moved[atom.index])
        else:
            mask = ''
        masks.append(mask)

    # Species info holds the information on the pseudopotential and
    # associated for each element
    if pseudopotentials is None:
        pseudopotentials = {}
    species_info = {}
    for species in set(atoms.get_chemical_symbols()):
        # Look in all possible locations for the pseudos and try to figure
        # out the number of valence electrons
        pseudo = pseudopotentials[species]
        species_info[species] = {'pseudo': pseudo}

    # Convert atoms into species.
    # Each different magnetic moment needs to be a separate type even with
    # the same pseudopotential (e.g. an up and a down for AFM).
    # if any magmom are > 0 or nspin == 2 then use species labels.
    # Rememeber: magnetisation uses 1 based indexes
    atomic_species = {}
    atomic_species_str = []
    atomic_positions_str = []

    nspin = input_parameters['system'].get('nspin', 1)  # 1 is the default
    noncolin = input_parameters['system'].get('noncolin', False)
    rescale_magmom_fac = kwargs.get('rescale_magmom_fac', 1.0)
    if any(atoms.get_initial_magnetic_moments()):
        if nspin == 1 and not noncolin:
            # Force spin on
            input_parameters['system']['nspin'] = 2
            nspin = 2

    if nspin == 2 or noncolin:
        # Magnetic calculation on
        for atom, mask, magmom in zip(
            atoms, masks, atoms.get_initial_magnetic_moments()
        ):
            if (atom.symbol, magmom) not in atomic_species:
                # for qe version 7.2 or older magmon must be rescale by
                # about a factor 10 to assume sensible values
                # since qe-v7.3 magmom values will be provided unscaled
                fspin = float(magmom) / rescale_magmom_fac
                # Index in the atomic species list
                sidx = len(atomic_species) + 1
                # Index for that atom type; no index for first one
                tidx = sum(atom.symbol == x[0] for x in atomic_species) or ' '
                atomic_species[(atom.symbol, magmom)] = (sidx, tidx)
                # Add magnetization to the input file
                mag_str = f'starting_magnetization({sidx})'
                input_parameters['system'][mag_str] = fspin
                species_pseudo = species_info[atom.symbol]['pseudo']
                atomic_species_str.append(
                    f'{atom.symbol}{tidx} {atom.mass} {species_pseudo}\n'
                )
            # lookup tidx to append to name
            sidx, tidx = atomic_species[(atom.symbol, magmom)]
            # construct line for atomic positions
            atomic_positions_str.append(
                format_atom_position(
                    atom, crystal_coordinates, mask=mask, tidx=tidx
                )
            )
    else:
        # Do nothing about magnetisation
        for atom, mask in zip(atoms, masks):
            if atom.symbol not in atomic_species:
                atomic_species[atom.symbol] = True  # just a placeholder
                species_pseudo = species_info[atom.symbol]['pseudo']
                atomic_species_str.append(
                    f'{atom.symbol} {atom.mass} {species_pseudo}\n'
                )
            # construct line for atomic positions
            atomic_positions_str.append(
                format_atom_position(atom, crystal_coordinates, mask=mask)
            )

    # Add computed parameters
    # different magnetisms means different types
    input_parameters['system']['ntyp'] = len(atomic_species)
    input_parameters['system']['nat'] = len(atoms)

    # Use cell as given or fit to a specific ibrav
    if 'ibrav' in input_parameters['system']:
        ibrav = input_parameters['system']['ibrav']
        if ibrav != 0:
            raise ValueError(ibrav_error_message)
    else:
        # Just use standard cell block
        input_parameters['system']['ibrav'] = 0

    # Construct input file into this
    pwi = input_parameters.to_string(list_form=True)

    # Pseudopotentials
    pwi.append('ATOMIC_SPECIES\n')
    pwi.extend(atomic_species_str)
    pwi.append('\n')

    # KPOINTS - add a MP grid as required
    if kspacing is not None:
        kgrid = kspacing_to_grid(atoms, kspacing)
    elif kpts is not None:
        if isinstance(kpts, dict) and 'path' not in kpts:
            kgrid, shift = kpts2sizeandoffsets(atoms=atoms, **kpts)
            koffset = []
            for i, x in enumerate(shift):
                assert x == 0 or abs(x * kgrid[i] - 0.5) < 1e-14
                koffset.append(0 if x == 0 else 1)
        else:
            kgrid = kpts
    else:
        kgrid = 'gamma'

    # True and False work here and will get converted by ':d' format
    if isinstance(koffset, int):
        koffset = (koffset,) * 3

    # BandPath object or bandpath-as-dictionary:
    if isinstance(kgrid, dict) or hasattr(kgrid, 'kpts'):
        pwi.append('K_POINTS crystal_b\n')
        assert hasattr(kgrid, 'path') or 'path' in kgrid
        kgrid = kpts2ndarray(kgrid, atoms=atoms)
        pwi.append(f'{len(kgrid)}\n')
        for k in kgrid:
            pwi.append(f'{k[0]:.14f} {k[1]:.14f} {k[2]:.14f} 0\n')
        pwi.append('\n')
    elif isinstance(kgrid, str) and (kgrid == 'gamma'):
        pwi.append('K_POINTS gamma\n')
        pwi.append('\n')
    elif isinstance(kgrid, np.ndarray):
        if np.shape(kgrid)[1] != 4:
            raise ValueError('Only Nx4 kgrids are supported right now.')
        pwi.append('K_POINTS crystal\n')
        pwi.append(f'{len(kgrid)}\n')
        for k in kgrid:
            pwi.append(f'{k[0]:.14f} {k[1]:.14f} {k[2]:.14f} {k[3]:.14f}\n')
        pwi.append('\n')
    else:
        pwi.append('K_POINTS automatic\n')
        pwi.append(
            f'{kgrid[0]} {kgrid[1]} {kgrid[2]} '
            f' {koffset[0]:d} {koffset[1]:d} {koffset[2]:d}\n'
        )
        pwi.append('\n')

    # CELL block, if required
    if input_parameters['SYSTEM']['ibrav'] == 0:
        pwi.append('CELL_PARAMETERS angstrom\n')
        pwi.append(
            '{cell[0][0]:.14f} {cell[0][1]:.14f} {cell[0][2]:.14f}\n'
            '{cell[1][0]:.14f} {cell[1][1]:.14f} {cell[1][2]:.14f}\n'
            '{cell[2][0]:.14f} {cell[2][1]:.14f} {cell[2][2]:.14f}\n'
            ''.format(cell=atoms.cell)
        )
        pwi.append('\n')

    # Positions - already constructed, but must appear after namelist
    if crystal_coordinates:
        pwi.append('ATOMIC_POSITIONS crystal\n')
    else:
        pwi.append('ATOMIC_POSITIONS angstrom\n')
    pwi.extend(atomic_positions_str)
    pwi.append('\n')

    # DONE!
    fd.write(''.join(pwi))

    if additional_cards:
        if isinstance(additional_cards, list):
            additional_cards = '\n'.join(additional_cards)
            additional_cards += '\n'

        fd.write(additional_cards)


def write_espresso_ph(
    fd, input_data=None, qpts=None, nat_todo_indices=None, **kwargs
) -> None:
    """
    Function that write the input file for a ph.x calculation. Normal namelist
    cards are passed in the input_data dictionary. Which can be either nested
    or flat, ASE style. The q-points are passed in the qpts list. If qplot is
    set to True then qpts is expected to be a list of list|tuple of length 4.
    Where the first three elements are the coordinates of the q-point in units
    of 2pi/alat and the last element is the weight of the q-point. if qplot is
    set to False then qpts is expected to be a simple list of length 4 (single
    q-point). Finally if ldisp is set to True, the above is discarded and the
    q-points are read from the nq1, nq2, nq3 cards in the input_data dictionary.

    Additionally, a nat_todo_indices kwargs (list[int]) can be specified in the
    kwargs. It will be used if nat_todo is set to True in the input_data
    dictionary.

    Globally, this function follows the convention set in the ph.x documentation
    (https://www.quantum-espresso.org/Doc/INPUT_PH.html)

    Parameters
    ----------
    fd
        The file descriptor of the input file.

    kwargs
        kwargs dictionary possibly containing the following keys:

        - input_data: dict
        - qpts: list[list[float]] | list[tuple[float]] | list[float]
        - nat_todo_indices: list[int]

    Returns
    -------
    None
    """

    input_data = Namelist(input_data)
    input_data.to_nested('ph', **kwargs)

    input_ph = input_data['inputph']

    inp_nat_todo = input_ph.get('nat_todo', 0)
    qpts = qpts or (0, 0, 0)

    pwi = input_data.to_string()

    fd.write(pwi)

    qplot = input_ph.get('qplot', False)
    ldisp = input_ph.get('ldisp', False)

    if qplot:
        fd.write(f'{len(qpts)}\n')
        for qpt in qpts:
            fd.write(f'{qpt[0]:0.8f} {qpt[1]:0.8f} {qpt[2]:0.8f} {qpt[3]:1d}\n')
    elif not (qplot or ldisp):
        fd.write(f'{qpts[0]:0.8f} {qpts[1]:0.8f} {qpts[2]:0.8f}\n')
    if inp_nat_todo:
        tmp = [str(i) for i in nat_todo_indices]
        fd.write(' '.join(tmp))
        fd.write('\n')


def read_espresso_ph(fileobj):
    """
    Function that reads the output of a ph.x calculation.
    It returns a dictionary where each q-point number is a key and
    the value is a dictionary with the following keys if available:

    - qpoints: The q-point in cartesian coordinates.
    - kpoints: The k-points in cartesian coordinates.
    - dieltensor: The dielectric tensor.
    - borneffcharge: The effective Born charges.
    - borneffcharge_dfpt: The effective Born charges from DFPT.
    - polarizability: The polarizability tensor.
    - modes: The phonon modes.
    - eqpoints: The symmetrically equivalent q-points.
    - freqs: The phonon frequencies.
    - mode_symmetries: The symmetries of the modes.
    - atoms: The atoms object.

    Some notes:

        - For some reason, the cell is not defined to high level of
          precision in ph.x outputs. Be careful when using the atoms object
          retrieved from this function.
        - This function can be called on incomplete calculations i.e.
          if the calculation couldn't diagonalize the dynamical matrix
          for some q-points, the results for the other q-points will
          still be returned.

    Parameters
    ----------
    fileobj
        The file descriptor of the output file.

    Returns
    -------
    dict
        The results dictionnary as described above.
    """
    QPOINTS = r'(?i)^\s*Calculation\s*of\s*q'
    NKPTS = r'(?i)^\s*number\s*of\s*k\s*points\s*'
    DIEL = r'(?i)^\s*Dielectric\s*constant\s*in\s*cartesian\s*axis\s*$'
    BORN = r'(?i)^\s*Effective\s*charges\s*\(d\s*Force\s*/\s*dE\)'
    POLA = r'(?i)^\s*Polarizability\s*(a.u.)\^3'
    REPR = r'(?i)^\s*There\s*are\s*\d+\s*irreducible\s*representations\s*$'
    EQPOINTS = r'(?i)^\s*Number\s*of\s*q\s*in\s*the\s*star\s*=\s*'
    DIAG = r'(?i)^\s*Diagonalizing\s*the\s*dynamical\s*matrix\s*$'
    MODE_SYM = r'(?i)^\s*Mode\s*symmetry,\s*'
    BORN_DFPT = r'(?i)^\s*Effective\s*charges\s*\(d\s*P\s*/\s*du\)'
    POSITIONS = r'(?i)^\s*site\s*n\..*\(alat\s*units\)'
    ALAT = r'(?i)^\s*celldm\(1\)='
    CELL = r'^\s*crystal\s*axes:\s*\(cart.\s*coord.\s*in\s*units\s*of\s*alat\)'
    ELECTRON_PHONON = r'(?i)^\s*electron-phonon\s*interaction\s*...\s*$'

    output = {
        QPOINTS: [],
        NKPTS: [],
        DIEL: [],
        BORN: [],
        BORN_DFPT: [],
        POLA: [],
        REPR: [],
        EQPOINTS: [],
        DIAG: [],
        MODE_SYM: [],
        POSITIONS: [],
        ALAT: [],
        CELL: [],
        ELECTRON_PHONON: [],
    }

    names = {
        QPOINTS: 'qpoints',
        NKPTS: 'kpoints',
        DIEL: 'dieltensor',
        BORN: 'borneffcharge',
        BORN_DFPT: 'borneffcharge_dfpt',
        POLA: 'polarizability',
        REPR: 'representations',
        EQPOINTS: 'eqpoints',
        DIAG: 'freqs',
        MODE_SYM: 'mode_symmetries',
        POSITIONS: 'positions',
        ALAT: 'alat',
        CELL: 'cell',
        ELECTRON_PHONON: 'ep_data',
    }

    unique = {
        QPOINTS: True,
        NKPTS: False,
        DIEL: True,
        BORN: True,
        BORN_DFPT: True,
        POLA: True,
        REPR: True,
        EQPOINTS: True,
        DIAG: True,
        MODE_SYM: True,
        POSITIONS: True,
        ALAT: True,
        CELL: True,
        ELECTRON_PHONON: True,
    }

    results = {}
    fdo_lines = [i for i in fileobj.read().splitlines() if i]
    n_lines = len(fdo_lines)

    for idx, line in enumerate(fdo_lines):
        for key in output:
            if bool(re.match(key, line)):
                output[key].append(idx)

    output = {key: np.array(value) for key, value in output.items()}

    def _read_qpoints(idx):
        match = re.findall(float_regex, fdo_lines[idx])
        return tuple(round(float(x), 7) for x in match)

    def _read_kpoints(idx):
        n_kpts = int(re.findall(float_regex, fdo_lines[idx])[0])
        kpts = []
        for line in fdo_lines[idx + 2 : idx + 2 + n_kpts]:
            if bool(re.search(r'^\s*k\(.*wk', line)):
                kpts.append(
                    [
                        round(float(x), 7)
                        for x in re.findall(float_regex, line)[1:]
                    ]
                )
        return np.array(kpts)

    def _read_repr(idx):
        n_repr, curr, n = int(re.findall(float_regex, fdo_lines[idx])[0]), 0, 0
        representations = {}
        while idx + n < n_lines:
            if re.search(r'^\s*Representation.*modes', fdo_lines[idx + n]):
                curr = int(re.findall(float_regex, fdo_lines[idx + n])[0])
                representations[curr] = {'done': False, 'modes': []}
            if re.search(
                r'Calculated\s*using\s*symmetry', fdo_lines[idx + n]
            ) or re.search(r'-\s*Done\s*$', fdo_lines[idx + n]):
                representations[curr]['done'] = True
            if re.search(r'(?i)^\s*(mode\s*#\s*\d\s*)+', fdo_lines[idx + n]):
                representations[curr]['modes'] = _read_modes(idx + n)
                if curr == n_repr:
                    break
            n += 1
        return representations

    def _read_modes(idx):
        n = 1
        n_modes = len(re.findall(r'mode', fdo_lines[idx]))
        modes = []
        while not modes or bool(re.match(r'^\s*\(', fdo_lines[idx + n])):
            tmp = re.findall(float_regex, fdo_lines[idx + n])
            modes.append([round(float(x), 5) for x in tmp])
            n += 1
        return np.hsplit(np.array(modes), n_modes)

    def _read_eqpoints(idx):
        n_star = int(re.findall(float_regex, fdo_lines[idx])[0])
        return np.loadtxt(
            fdo_lines[idx + 2 : idx + 2 + n_star], usecols=(1, 2, 3)
        ).reshape(-1, 3)

    def _read_freqs(idx):
        n = 0
        freqs = []
        stop = 0
        while not freqs or stop < 2:
            if bool(re.search(r'^\s*freq', fdo_lines[idx + n])):
                tmp = re.findall(float_regex, fdo_lines[idx + n])[1]
                freqs.append(float(tmp))
            if bool(re.search(r'\*{5,}', fdo_lines[idx + n])):
                stop += 1
            n += 1
        return np.array(freqs)

    def _read_sym(idx):
        n = 1
        sym = {}
        while bool(re.match(r'^\s*freq', fdo_lines[idx + n])):
            r = re.findall('\\d+', fdo_lines[idx + n])
            r = tuple(range(int(r[0]), int(r[1]) + 1))
            sym[r] = fdo_lines[idx + n].split('-->')[1].strip()
            sym[r] = re.sub(r'\s+', ' ', sym[r])
            n += 1
        return sym

    def _read_epsil(idx):
        epsil = np.zeros((3, 3))
        for n in range(1, 4):
            tmp = re.findall(float_regex, fdo_lines[idx + n])
            epsil[n - 1] = [round(float(x), 9) for x in tmp]
        return epsil

    def _read_born(idx):
        n = 1
        born = []
        while idx + n < n_lines:
            if re.search(r'^\s*atom\s*\d\s*\S', fdo_lines[idx + n]):
                pass
            elif re.search(r'^\s*E\*?(x|y|z)\s*\(', fdo_lines[idx + n]):
                tmp = re.findall(float_regex, fdo_lines[idx + n])
                born.append([round(float(x), 5) for x in tmp])
            else:
                break
            n += 1
        born = np.array(born)
        return np.vsplit(born, len(born) // 3)

    def _read_born_dfpt(idx):
        n = 1
        born = []
        while idx + n < n_lines:
            if re.search(r'^\s*atom\s*\d\s*\S', fdo_lines[idx + n]):
                pass
            elif re.search(r'^\s*P(x|y|z)\s*\(', fdo_lines[idx + n]):
                tmp = re.findall(float_regex, fdo_lines[idx + n])
                born.append([round(float(x), 5) for x in tmp])
            else:
                break
            n += 1
        born = np.array(born)
        return np.vsplit(born, len(born) // 3)

    def _read_pola(idx):
        pola = np.zeros((3, 3))
        for n in range(1, 4):
            tmp = re.findall(float_regex, fdo_lines[idx + n])[:3]
            pola[n - 1] = [round(float(x), 2) for x in tmp]
        return pola

    def _read_positions(idx):
        positions = []
        symbols = []
        n = 1
        while re.findall(r'^\s*\d+', fdo_lines[idx + n]):
            symbols.append(fdo_lines[idx + n].split()[1])
            positions.append(
                [
                    round(float(x), 5)
                    for x in re.findall(float_regex, fdo_lines[idx + n])[-3:]
                ]
            )
            n += 1
        atoms = Atoms(positions=positions, symbols=symbols)
        atoms.pbc = True
        return atoms

    def _read_alat(idx):
        return round(float(re.findall(float_regex, fdo_lines[idx])[1]), 5)

    def _read_cell(idx):
        cell = []
        n = 1
        while re.findall(r'^\s*a\(\d\)', fdo_lines[idx + n]):
            cell.append(
                [
                    round(float(x), 4)
                    for x in re.findall(float_regex, fdo_lines[idx + n])[-3:]
                ]
            )
            n += 1
        return np.array(cell)

    def _read_electron_phonon(idx):
        results = {}

        broad_re = r'^\s*Gaussian\s*Broadening:\s+([\d.]+)\s+Ry, ngauss=\s+\d+'
        dos_re = (
            r'^\s*DOS\s*=\s*([\d.]+)\s*states/'
            r'spin/Ry/Unit\s*Cell\s*at\s*Ef=\s+([\d.]+)\s+eV'
        )
        lg_re = r'^\s*lambda\(\s+(\d+)\)=\s+([\d.]+)\s+gamma=\s+([\d.]+)\s+GHz'
        end_re = r'^\s*Number\s*of\s*q\s*in\s*the\s*star\s*=\s+(\d+)$'

        lambdas = []
        gammas = []

        current = None

        n = 1
        while idx + n < n_lines:
            line = fdo_lines[idx + n]

            broad_match = re.match(broad_re, line)
            dos_match = re.match(dos_re, line)
            lg_match = re.match(lg_re, line)
            end_match = re.match(end_re, line)

            if broad_match:
                if lambdas:
                    results[current]['lambdas'] = lambdas
                    results[current]['gammas'] = gammas
                    lambdas = []
                    gammas = []
                current = float(broad_match[1])
                results[current] = {}
            elif dos_match:
                results[current]['dos'] = float(dos_match[1])
                results[current]['fermi'] = float(dos_match[2])
            elif lg_match:
                lambdas.append(float(lg_match[2]))
                gammas.append(float(lg_match[3]))

            if end_match:
                results[current]['lambdas'] = lambdas
                results[current]['gammas'] = gammas
                break

            n += 1

        return results

    properties = {
        NKPTS: _read_kpoints,
        DIEL: _read_epsil,
        BORN: _read_born,
        BORN_DFPT: _read_born_dfpt,
        POLA: _read_pola,
        REPR: _read_repr,
        EQPOINTS: _read_eqpoints,
        DIAG: _read_freqs,
        MODE_SYM: _read_sym,
        POSITIONS: _read_positions,
        ALAT: _read_alat,
        CELL: _read_cell,
        ELECTRON_PHONON: _read_electron_phonon,
    }

    iblocks = np.append(output[QPOINTS], n_lines)

    for qnum, (past, future) in enumerate(zip(iblocks[:-1], iblocks[1:])):
        qpoint = _read_qpoints(past)
        results[qnum + 1] = curr_result = {'qpoint': qpoint}
        for prop in properties:
            p = (past < output[prop]) & (output[prop] < future)
            selected = output[prop][p]
            if len(selected) == 0:
                continue
            if unique[prop]:
                idx = output[prop][p][-1]
                curr_result[names[prop]] = properties[prop](idx)
            else:
                tmp = {k + 1: 0 for k in range(len(selected))}
                for k, idx in enumerate(selected):
                    tmp[k + 1] = properties[prop](idx)
                curr_result[names[prop]] = tmp
        alat = curr_result.pop('alat', 1.0)
        atoms = curr_result.pop('positions', None)
        cell = curr_result.pop('cell', np.eye(3))
        if atoms:
            atoms.positions *= alat * units['Bohr']
            atoms.cell = cell * alat * units['Bohr']
            atoms.wrap()
            curr_result['atoms'] = atoms

    return results


def write_fortran_namelist(
    fd, input_data=None, binary=None, additional_cards=None, **kwargs
) -> None:
    """
    Function which writes input for simple espresso binaries.
    List of supported binaries are in the espresso_keys.py file.
    Non-exhaustive list (to complete)

    Note: "EOF" is appended at the end of the file.
    (https://lists.quantum-espresso.org/pipermail/users/2020-November/046269.html)

    Additional fields are written 'as is' in the input file. It is expected
    to be a string or a list of strings.

    Parameters
    ----------
    fd
        The file descriptor of the input file.
    input_data: dict
        A flat or nested dictionary with input parameters for the binary.x
    binary: str
        Name of the binary
    additional_cards: str | list[str]
        Additional fields to be written at the end of the input file, after
        the namelist. It is expected to be a string or a list of strings.

    Returns
    -------
    None
    """
    input_data = Namelist(input_data)

    if binary:
        input_data.to_nested(binary, **kwargs)

    pwi = input_data.to_string()

    fd.write(pwi)

    if additional_cards:
        if isinstance(additional_cards, list):
            additional_cards = '\n'.join(additional_cards)
            additional_cards += '\n'

        fd.write(additional_cards)

    fd.write('EOF')


@deprecated('Please use the ase.io.espresso.Namelist class', DeprecationWarning)
def construct_namelist(parameters=None, keys=None, warn=False, **kwargs):
    """
    Construct an ordered Namelist containing all the parameters given (as
    a dictionary or kwargs). Keys will be inserted into their appropriate
    section in the namelist and the dictionary may contain flat and nested
    structures. Any kwargs that match input keys will be incorporated into
    their correct section. All matches are case-insensitive, and returned
    Namelist object is a case-insensitive dict.

    If a key is not known to ase, but in a section within `parameters`,
    it will be assumed that it was put there on purpose and included
    in the output namelist. Anything not in a section will be ignored (set
    `warn` to True to see ignored keys).

    Keys with a dimension (e.g. Hubbard_U(1)) will be incorporated as-is
    so the `i` should be made to match the output.

    The priority of the keys is:
        kwargs[key] > parameters[key] > parameters[section][key]
    Only the highest priority item will be included.

    .. deprecated:: 3.23.0
        Please use :class:`ase.io.espresso.Namelist` instead.

    Parameters
    ----------
    parameters: dict
        Flat or nested set of input parameters.
    keys: Namelist | dict
        Namelist to use as a template for the output.
    warn: bool
        Enable warnings for unused keys.

    Returns
    -------
    input_namelist: Namelist
        pw.x compatible namelist of input parameters.

    """

    if keys is None:
        keys = deepcopy(pw_keys)
    # Convert everything to Namelist early to make case-insensitive
    if parameters is None:
        parameters = Namelist()
    else:
        # Maximum one level of nested dict
        # Don't modify in place
        parameters_namelist = Namelist()
        for key, value in parameters.items():
            if isinstance(value, dict):
                parameters_namelist[key] = Namelist(value)
            else:
                parameters_namelist[key] = value
        parameters = parameters_namelist

    # Just a dict
    kwargs = Namelist(kwargs)

    # Final parameter set
    input_namelist = Namelist()

    # Collect
    for section in keys:
        sec_list = Namelist()
        for key in keys[section]:
            # Check all three separately and pop them all so that
            # we can check for missing values later
            value = None

            if key in parameters.get(section, {}):
                value = parameters[section].pop(key)
            if key in parameters:
                value = parameters.pop(key)
            if key in kwargs:
                value = kwargs.pop(key)

            if value is not None:
                sec_list[key] = value

            # Check if there is a key(i) version (no extra parsing)
            for arg_key in list(parameters.get(section, {})):
                if arg_key.split('(')[0].strip().lower() == key.lower():
                    sec_list[arg_key] = parameters[section].pop(arg_key)
            cp_parameters = parameters.copy()
            for arg_key in cp_parameters:
                if arg_key.split('(')[0].strip().lower() == key.lower():
                    sec_list[arg_key] = parameters.pop(arg_key)
            cp_kwargs = kwargs.copy()
            for arg_key in cp_kwargs:
                if arg_key.split('(')[0].strip().lower() == key.lower():
                    sec_list[arg_key] = kwargs.pop(arg_key)

        # Add to output
        input_namelist[section] = sec_list

    unused_keys = list(kwargs)
    # pass anything else already in a section
    for key, value in parameters.items():
        if key in keys and isinstance(value, dict):
            input_namelist[key].update(value)
        elif isinstance(value, dict):
            unused_keys.extend(list(value))
        else:
            unused_keys.append(key)

    if warn and unused_keys:
        warnings.warn('Unused keys: {}'.format(', '.join(unused_keys)))

    return input_namelist


@deprecated(
    'Please use the .to_string() method of Namelist instead.',
    DeprecationWarning,
)
def namelist_to_string(input_parameters):
    """Format a Namelist object as a string for writing to a file.
    Assume sections are ordered (taken care of in namelist construction)
    and that repr converts to a QE readable representation (except bools)

    .. deprecated:: 3.23.0
        Please use the :meth:`ase.io.espresso.Namelist.to_string` method
        instead.

    Parameters
    ----------
    input_parameters : Namelist | dict
        Expecting a nested dictionary of sections and key-value data.

    Returns
    -------
    pwi : List[str]
        Input line for the namelist
    """
    pwi = []
    for section in input_parameters:
        pwi.append(f'&{section.upper()}\n')
        for key, value in input_parameters[section].items():
            if value is True:
                pwi.append(f'   {key:16} = .true.\n')
            elif value is False:
                pwi.append(f'   {key:16} = .false.\n')
            elif isinstance(value, Path):
                pwi.append(f'   {key:16} = "{value}"\n')
            else:
                # repr format to get quotes around strings
                pwi.append(f'   {key:16} = {value!r}\n')
        pwi.append('/\n')  # terminate section
    pwi.append('\n')
    return pwi
