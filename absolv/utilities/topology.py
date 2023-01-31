"""Utilities for manipulating OpenFF topology objects."""
from typing import List, Set, Tuple

from openff.toolkit.topology import Topology


def topology_to_components(topology: Topology) -> List[Tuple[str, int]]:
    """A helper method for condensing a topology down to a list of components
    and their counts.

    Notes:
        If the topology is not contiguous then the returned list may contain multiple
        tuples with the same smiles but different counts.

    Args:
        topology: The topology to condense.

    Returns:
        A list of the form ``components[i] = (smiles_i, count_i)`` where
        ``smiles_i`` is the SMILES representation of component `i` and
        ``count_i`` is the number of corresponding instances of that component
        to create.
    """
    components = []

    current_smiles = None
    current_count = 0

    for topology_molecule in topology.topology_molecules:

        smiles = topology_molecule.reference_molecule.to_smiles()

        if smiles == current_smiles:

            current_count += 1
            continue

        if current_count > 0:
            components.append((current_smiles, current_count))

        current_smiles = smiles
        current_count = 1

    if current_count > 0:
        components.append((current_smiles, current_count))

    return components


def topology_to_atom_indices(topology: Topology) -> List[Set[int]]:
    """A helper method for extracting the sets of atom indices associated with each
    molecule in a topology.

    Args:
        topology: The topology to extract the atom indices from.

    Returns:
        The set of atoms indices associated with each molecule in the topology.
    """

    atom_indices: List[Set[int]] = []
    current_atom_index = 0

    for topology_molecule in topology.topology_molecules:

        atom_indices.append(
            {i + current_atom_index for i in range(topology_molecule.n_atoms)}
        )
        current_atom_index += topology_molecule.n_atoms

    return atom_indices
