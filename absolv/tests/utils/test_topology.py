import openff.toolkit
import pytest

from absolv.utils.topology import topology_to_components


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
