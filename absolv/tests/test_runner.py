# class TestBaseRunner(BaseTemporaryDirTest):
#     def test_load_solvent_inputs(self, argon_force_field):
#         os.makedirs("test-dir")
#
#         with temporary_cd("test-dir"):
#             BaseRunner._setup_solvent(
#                 "solvent-a", [("[Ar]", 10)], argon_force_field, 1, 9
#             )
#
#         (
#             topology,
#             coords,
#             chemical_system,
#             alchemical_system,
#         ) = BaseRunner._load_solvent_inputs("test-dir")
#
#         assert isinstance(topology, Topology)
#
#         assert isinstance(coords, unit.Quantity)
#         assert coords.shape == (10, 3)
#
#         assert isinstance(chemical_system, openmm.System)
#         assert "lambda" not in openmm.XmlSerializer.serialize(chemical_system)
#
#         assert isinstance(alchemical_system, openmm.System)
#         assert "lambda" in openmm.XmlSerializer.serialize(alchemical_system)
#
#     @pytest.mark.parametrize(
#         "force_field",
#         [
#             ForceField("openff-2.0.0.offxml"),
#             lambda topology, *_: ForceField("openff-2.0.0.offxml").
#             create_openmm_system(
#                 topology
#             ),
#         ],
#     )
#     def test_setup_solvent(self, force_field):
#         BaseRunner._setup_solvent(
#             "solvent-a", [("CO", 1), ("O", 10)], force_field, 1, 10
#         )
#
#         for expected_file in [
#             "coords-initial.pdb",
#             "coords-initial.npy",
#             "system-chemical.xml",
#             "system-alchemical.xml",
#             "topology.pkl",
#         ]:
#             assert os.path.isfile(expected_file)
#
#         pdb_file = PDBFile("coords-initial.pdb")
#
#         assert all_close(
#             numpy.load("coords-initial.npy") * unit.angstrom,
#             numpy.array(
#                 [
#                     [value.value_in_unit(unit.nanometers) for value in coord]
#                     for coord in pdb_file.positions
#                 ]
#             )
#             * unit.nanometers,
#             atol=1.0e-1,
#         )
#
#         with open("topology.pkl", "rb") as file:
#             topology: Topology = pickle.load(file)
#
#         assert topology.n_topology_molecules == 11
#
#         with open("system-chemical.xml") as file:
#             contents = file.read()
#
#             assert "lambda_sterics" not in contents
#             openmm.XmlSerializer.deserialize(contents)
#
#         with open("system-alchemical.xml") as file:
#             contents = file.read()
#
#             assert "lambda_sterics" in contents
#             openmm.XmlSerializer.deserialize(contents)
#
#     def test_setup(self, argon_eq_schema, argon_force_field):
#         BaseRunner.setup(argon_eq_schema, argon_force_field, "argon-dir")
#
#         assert os.path.isdir("argon-dir")
#
#         for solvent_index in ("solvent-a", "solvent-b"):
#             assert os.path.isdir(os.path.join("argon-dir", solvent_index))
#             assert os.path.isfile(
#                 os.path.join("argon-dir", solvent_index, "topology.pkl")
#             )
#
#         with open(os.path.join("argon-dir", "schema.json")) as file:
#             assert argon_eq_schema.json(indent=4) == file.read()
#
#     @pytest.mark.parametrize("sampler", ["independent", "hremd"])
#     def test_run_solvent(
#         self, sampler, argon_eq_schema, argon_force_field, monkeypatch
#     ):
#         argon_eq_schema.alchemical_protocol_a.sampler = sampler
#         argon_eq_schema.alchemical_protocol_b.sampler = sampler
#
#         run_class = None
#         run_directories = []
#
#         def mock_run(self, directory):
#             nonlocal run_class
#             run_class = type(self)
#
#             run_directories.append(directory)
#
#         monkeypatch.setattr(AlchemicalOpenMMSimulation, "run", mock_run)
#         monkeypatch.setattr(RepexAlchemicalOpenMMSimulation, "run", mock_run)
#
#         BaseRunner._setup_solvent(
#             "solvent-a", [("[Ar]", 128)], argon_force_field, 1, 127
#         )
#         BaseRunner._run_solvent(
#             argon_eq_schema.alchemical_protocol_a,
#             State(temperature=88.5, pressure=1.0),
#             "Reference",
#             states=[0, 2] if sampler != "hremd" else None,
#         )
#
#         assert run_directories == (
#             ["state-0", "state-2"] if sampler != "hremd" else [""]
#         )
#
#         expected_run_class = (
#             AlchemicalOpenMMSimulation
#             if sampler != "hremd"
#             else RepexAlchemicalOpenMMSimulation
#         )
#         assert expected_run_class == run_class
#
#     def test_run_solvent_state_error(self, argon_eq_schema, argon_force_field):
#         argon_eq_schema.alchemical_protocol_a.sampler = "hremd"
#         argon_eq_schema.alchemical_protocol_b.sampler = "hremd"
#
#         BaseRunner._setup_solvent(
#             "solvent-a", [("[Ar]", 128)], argon_force_field, 1, 127
#         )
#
#         with pytest.raises(NotImplementedError, match="All lambda states must be
#         run"):
#             BaseRunner._run_solvent(
#                 argon_eq_schema.alchemical_protocol_a,
#                 State(temperature=88.5, pressure=1.0),
#                 "Reference",
#                 states=[0, 2],
#             )
#
#
# class TestEquilibriumRunner(BaseTemporaryDirTest):
#     def test_run(self, argon_eq_schema, argon_force_field):
#         EquilibriumRunner.setup(argon_eq_schema, argon_force_field, "test-dir")
#         EquilibriumRunner.run("test-dir", platform="Reference")
#
#         assert all(
#             os.path.isdir(os.path.join("test-dir", solvent_index, "state-0"))
#             for solvent_index in ("solvent-a", "solvent-b")
#         )
#         assert all(
#             os.path.isfile(
#                 os.path.join(
#                     "test-dir", solvent_index, "state-0", "production-trajectory.dcd"
#                 )
#             )
#             for solvent_index in ("solvent-a", "solvent-b")
#         )
#
#     def test_analyze(self, argon_eq_schema):
#         with open("schema.json", "w") as file:
#             file.write(argon_eq_schema.json())
#
#         state_indices = {"solvent-a": [0, 1, 2], "solvent-b": [0, 1]}
#
#         for solvent_index in state_indices:
#             for i in state_indices[solvent_index]:
#                 os.makedirs(os.path.join(solvent_index, f"state-{i}"))
#
#                 numpy.savetxt(
#                     os.path.join(solvent_index, f"state-{i}",
#                     "lambda-potentials.csv"),
#                     numpy.random.random((1, len(state_indices[solvent_index]))),
#                 )
#
#         result = EquilibriumRunner.analyze("")
#
#         assert isinstance(result, Result)
#         assert result.input_schema.json() == argon_eq_schema.json()
#
#
# class TestNonEquilibriumRunner(BaseTemporaryDirTest):
#     @pytest.fixture(autouse=True)
#     def _setup_argon(self, _temporary_cd, argon_force_field):
#         for solvent_index, n_particles, cell_lengths, cell_angles in (
#             ("solvent-a", 128, numpy.ones((2, 3)) * 10.0, numpy.ones((2, 3)) * 90.0),
#             ("solvent-b", 1, None, None),
#         ):
#             os.makedirs(solvent_index)
#
#             with temporary_cd(solvent_index):
#                 NonEquilibriumRunner._setup_solvent(
#                     solvent_index,
#                     [("[Ar]", n_particles)],
#                     argon_force_field,
#                     1,
#                     n_particles - 1,
#                 )
#
#                 with open("topology.pkl", "rb") as file:
#                     topology = pickle.load(file).to_openmm()
#
#                 trajectory = mdtraj.Trajectory(
#                     xyz=numpy.zeros((2, n_particles, 3)),
#                     topology=mdtraj.Topology.from_openmm(topology),
#                     unitcell_lengths=cell_lengths,
#                     unitcell_angles=cell_angles,
#                 )
#
#                 for state_index in (0, 1):
#                     os.makedirs(f"state-{state_index}")
#                     trajectory.save(
#                         os.path.join(
#                             f"state-{state_index}", "production-trajectory.dcd"
#                         )
#                     )
#
#     def test_run_switching_checkpoint(self, argon_neq_schema):
#         expected_forward_work = numpy.array([1.0, 2.0, 3.0])
#         expected_reverse_work = numpy.array([3.0, 2.0, 1.0])
#
#         with temporary_cd("solvent-a"):
#             numpy.savetxt("forward-work.csv", expected_forward_work, delimiter=" ")
#             numpy.savetxt("reverse-work.csv", expected_reverse_work, delimiter=" ")
#
#             forward_work, reverse_work = NonEquilibriumRunner._run_switching(
#                 argon_neq_schema.alchemical_protocol_a,
#                 argon_neq_schema.state,
#                 "Reference",
#             )
#
#         assert numpy.allclose(forward_work, expected_forward_work)
#         assert numpy.allclose(reverse_work, expected_reverse_work)
#
#     def test_run(self, argon_neq_schema, monkeypatch):
#         argon_neq_schema = copy.deepcopy(argon_neq_schema)
#
#         monkeypatch.setattr(
#             NonEquilibriumRunner, "_run_solvent", lambda *args, **kwargs: None
#         )
#         monkeypatch.setattr(
#             NonEquilibriumOpenMMSimulation,
#             "run",
#             lambda *args, **kwargs: (numpy.random.random(), numpy.random.random()),
#         )
#
#         with open("schema.json", "w") as file:
#             file.write(argon_neq_schema.json())
#
#         NonEquilibriumRunner.run("", "Reference")
#
#         for solvent_index in ("solvent-a", "solvent-b"):
#             forward_work = numpy.genfromtxt(
#                 os.path.join(solvent_index, "forward-work.csv"), delimiter=" "
#             )
#             reverse_work = numpy.genfromtxt(
#                 os.path.join(solvent_index, "reverse-work.csv"), delimiter=" "
#             )
#
#             assert forward_work.shape == (2,)
#             assert not numpy.allclose(forward_work, 0.0)
#
#             assert reverse_work.shape == (2,)
#             assert not numpy.allclose(reverse_work, 0.0)
#
#     def test_analyze(self, argon_eq_schema):
#         with open("schema.json", "w") as file:
#             file.write(argon_eq_schema.json())
#
#         expected_forward_work = numpy.array([1.0, 2.0, 3.0])
#         expected_reverse_work = numpy.array([3.0, 2.0, 1.0])
#
#         for solvent_index in ("solvent-a", "solvent-b"):
#             numpy.savetxt(
#                 os.path.join(solvent_index, "forward-work.csv"),
#                 expected_forward_work,
#                 delimiter=" ",
#             )
#             numpy.savetxt(
#                 os.path.join(solvent_index, "reverse-work.csv"),
#                 expected_reverse_work,
#                 delimiter=" ",
#             )
#
#         result = NonEquilibriumRunner.analyze("")
#
#         assert isinstance(result, Result)
#         assert result.input_schema.json() == argon_eq_schema.json()
