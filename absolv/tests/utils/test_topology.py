import pytest
import openff.toolkit

from absolv.utils.topology import topology_to_atom_indices, topology_to_components


@pytest.mark.parametrize("n_counts", [[3, 1, 2], [3, 1, 1]])
def test_topology_to_components(n_counts):
    topology = openff.toolkit.Topology.from_molecules(
        [openff.toolkit.Molecule.from_smiles("O")] * n_counts[0]
        + [openff.toolkit.Molecule.from_smiles("C")] * n_counts[1]
        + [openff.toolkit.Molecule.from_smiles("O")] * n_counts[2]
    )

    components = topology_to_components(topology)

    assert components == [
        ("[H][O][H]", n_counts[0]),
        ("[H][C]([H])([H])[H]", n_counts[1]),
        ("[H][O][H]", n_counts[2]),
    ]


def test_topology_to_atom_indices():
    topology = openff.toolkit.Topology.from_molecules(
        [openff.toolkit.Molecule.from_smiles("O")] * 1
        + [openff.toolkit.Molecule.from_smiles("C")] * 2
        + [openff.toolkit.Molecule.from_smiles("O")] * 3
    )

    atom_indices = topology_to_atom_indices(topology)

    assert atom_indices == [
        {0, 1, 2},
        {3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12},
        {13, 14, 15},
        {16, 17, 18},
        {19, 20, 21},
    ]
