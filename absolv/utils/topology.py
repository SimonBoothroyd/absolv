"""Utilities for manipulating OpenFF topology objects."""
import openff.toolkit


def topology_to_components(topology: openff.toolkit.Topology) -> list[tuple[str, int]]:
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

    for molecule in topology.molecules:
        smiles = molecule.to_smiles()

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


def topology_to_atom_indices(topology: openff.toolkit.Topology) -> list[set[int]]:
    """A helper method for extracting the sets of atom indices associated with each
    molecule in a topology.

    Args:
        topology: The topology to extract the atom indices from.

    Returns:
        The set of atoms indices associated with each molecule in the topology.
    """

    atom_indices: list[set[int]] = []
    current_atom_idx = 0

    for molecule in topology.molecules:
        atom_indices.append({i + current_atom_idx for i in range(molecule.n_atoms)})
        current_atom_idx += molecule.n_atoms

    return atom_indices
