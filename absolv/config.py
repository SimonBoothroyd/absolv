"""Configure free energy calculations."""
import math
import typing

import femto.md.config
from femto.md.utils.models import OpenMMQuantity
import pydantic
import openmm.unit

KCAL_MOL = openmm.unit.kilocalories_per_mole

# fmt: off
DEFAULT_LAMBDA_ELECTROSTATICS_VACUUM = [1.0, 0.75, 0.5, 0.25, 0.0]
DEFAULT_LAMBDA_ELECTROSTATICS_SOLVENT = [
    1.00, 0.75, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
]

DEFAULT_LAMBDA_STERICS_VACUUM = [1.0, 1.0, 1.0, 1.0, 1.0]
DEFAULT_LAMBDA_STERICS_SOLVENT = [
    1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50,
    0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00,
]
# fmt: on


class System(pydantic.BaseModel):
    """Define the two solvents that solutes will be transferred between (a -> b),
    as well as the solutes themselves.
    """

    solutes: dict[str, pydantic.PositiveInt] = pydantic.Field(
        ...,
        description="A dictionary containing the SMILES patterns of each solute in the "
        "system as well as how many instances of each there should be.",
        min_length=1,
    )

    solvent_a: dict[str, pydantic.PositiveInt] | None = pydantic.Field(
        ...,
        description="A dictionary containing the SMILES patterns of each component "
        "in the first solvent as well as how many instances of each there should be."
        "A value of ``None`` should be used to indicate vacuum.",
    )
    solvent_b: dict[str, pydantic.PositiveInt] | None = pydantic.Field(
        ...,
        description="A dictionary containing the SMILES patterns of each component "
        "in the second solvent as well as how many instances of each there should be. "
        "A value of ``None`` should be used to indicate vacuum.",
    )

    @property
    def n_solute_molecules(self) -> int:
        """Returns the total number of solute molecules that will be present."""
        return sum(self.solutes.values())

    @property
    def n_solvent_molecules_a(self) -> int:
        """Returns the total number of solvent molecules that will be present in the
        first solution."""
        return 0 if self.solvent_a is None else sum(self.solvent_a.values())

    @property
    def n_solvent_molecules_b(self) -> int:
        """Returns the total number of solvent molecules that will be present in the
        second solution."""
        return 0 if self.solvent_b is None else sum(self.solvent_b.values())

    @pydantic.field_validator("solvent_a")
    def _validate_solvent_a(cls, value):
        if value is None:
            return value

        assert (
            len(value) > 0
        ), "at least one solvent must be specified when `solvent_a` is not none"
        return value

    @pydantic.field_validator("solvent_b")
    def _validate_solvent_b(cls, value):
        if value is None:
            return value

        assert (
            len(value) > 0
        ), "at least one solvent must be specified when `solvent_b` is not none"
        return value

    def to_components(self) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
        """Converts this object into two lists: one containing the identities and
        counts of the molecules present in the first solution, and one containing the
        same for the second solution.

        The identity and amount are stored in a tuple as a SMILES pattern and integer
        count.
        """

        components_a = [*self.solutes.items()] + (
            [] if self.solvent_a is None else [*self.solvent_a.items()]
        )
        components_b = [*self.solutes.items()] + (
            [] if self.solvent_b is None else [*self.solvent_b.items()]
        )

        return components_a, components_b


class MinimizationProtocol(pydantic.BaseModel):
    """Configure how a system should be energy minimized."""

    tolerance: OpenMMQuantity[
        openmm.unit.kilojoule_per_mole / openmm.unit.nanometers
    ] = pydantic.Field(
        10.0 * openmm.unit.kilojoule_per_mole / openmm.unit.nanometers,
        description="How precisely the energy minimum must be located [kj / mol / nm]",
    )
    max_iterations: pydantic.NonNegativeInt = pydantic.Field(
        0,
        description="The maximum number of iterations to perform. If this is 0, "
        "minimization is continued until the results converge.",
    )


class SimulationProtocol(pydantic.BaseModel):
    """Configure how a system should be evolved by molecular simulation."""

    integrator: femto.md.config.LangevinIntegrator = pydantic.Field(
        femto.md.config.LangevinIntegrator(
            timestep=2.0 * openmm.unit.femtosecond,
            friction=1.0 / openmm.unit.picosecond,
        ),
        description="The integrator to use for the simulation.",
    )
    n_steps: pydantic.PositiveInt = pydantic.Field(
        ...,
        description="The number of steps to evolve the system by.",
    )


class HREMDProtocol(pydantic.BaseModel):
    """Configure how a system should be evolved by Hamiltonian replica exchange."""

    integrator: femto.md.config.LangevinIntegrator = pydantic.Field(
        femto.md.config.LangevinIntegrator(
            timestep=2.0 * openmm.unit.femtosecond,
            friction=1.0 / openmm.unit.picosecond,
        ),
        description="The integrator to use for the simulation.",
    )

    n_warmup_steps: int = pydantic.Field(
        50000,
        description="The number of steps to run each replica for before starting hremd "
        "trials. All energies gathered during this period will be discarded.",
    )

    n_steps_per_cycle: int = pydantic.Field(
        1000,
        description="The number of steps to propagate the system by before attempting "
        "an exchange.",
    )
    n_cycles: int = pydantic.Field(
        2500,
        description="The number of cycles of "
        "'propagate the system' -> 'exchange replicas' to run.",
    )


class EquilibriumProtocol(pydantic.BaseModel):
    """Configure how an equilibrium (e.g. TI, MBAR) alchemical free energy
    calculation."""

    type: typing.Literal["equilibrium"] = "equilibrium"

    minimization_protocol: MinimizationProtocol = pydantic.Field(
        MinimizationProtocol(),
        description="Whether to minimize the energy of the system prior to any "
        "simulations.",
    )

    equilibration_protocol: SimulationProtocol = pydantic.Field(
        SimulationProtocol(n_steps=100000),  # 200 ps
        description="The (optional) protocol that describes the equilibration "
        "simulation to run prior to the production one.",
    )
    production_protocol: HREMDProtocol = pydantic.Field(
        HREMDProtocol(n_steps_per_cycle=6250, n_cycles=160),  # 2 ns
        description="The protocol that describes the production to run.",
    )

    lambda_sterics: list[pydantic.confloat(ge=0.0, le=1.0)] = pydantic.Field(
        ...,
        description="The alchemical pathway to transform the vdW interactions along. A "
        "value of 1.0 represents a fully interacting system while a value of 0.0 "
        "represents a system with the solute-solute and solute-solvent vdW "
        "interactions disabled.",
    )
    lambda_electrostatics: list[pydantic.confloat(ge=0.0, le=1.0)] = pydantic.Field(
        ...,
        description="The alchemical pathway to transform the electrostatic "
        "interactions along. A value of 1.0 represents a fully interacting system "
        "while a value of 0.0 represents a system with the solute-solute and "
        "solute-solvent electrostatic interactions disabled.",
    )

    @property
    def n_states(self) -> int:
        """Returns the number of lambda states that will be sampled at."""
        return len(self.lambda_sterics)

    @pydantic.model_validator(mode="after")
    def _validate_lambda_lengths(self) -> "EquilibriumProtocol":
        lambda_lists = [self.lambda_sterics, self.lambda_electrostatics]

        assert all(
            len(lambda_list) == len(lambda_lists[0]) for lambda_list in lambda_lists
        ), "lambda lists must be the same length"

        return self


class SwitchingProtocol(pydantic.BaseModel):
    """Configure non-reversibly driving a system between to alchemical states."""

    n_electrostatic_steps: pydantic.NonNegativeInt = pydantic.Field(
        ...,
        description="The number of steps to annihilate the electrostatics interactions "
        "over. The total time needed to annihilate the electrostatics interactions "
        "will be ``n_electrostatic_steps * n_steps_per_electrostatic_step * timestep``",
    )
    n_steps_per_electrostatic_step: pydantic.NonNegativeInt = pydantic.Field(
        ...,
        description="The number of timesteps to evolve the system by each time the "
        "electrostatics interactions are modified. A value of 1 will give a 'smooth' "
        "transition between the each discrete lambda value whereas a value greater "
        "than 1 will yield a stepwise transition as shown in Figure 3 of "
        "doi:10.1063/1.4712028.",
    )

    n_steric_steps: pydantic.NonNegativeInt = pydantic.Field(
        ...,
        description="The number of steps to decouple the sterics interactions over "
        "once the electrostatics interactions have been annihilated. The total time "
        "needed to annihilate the sterics interactions will be "
        "``n_steric_steps * n_steps_per_steric_step * timestep``",
    )
    n_steps_per_steric_step: pydantic.NonNegativeInt = pydantic.Field(
        ...,
        description="The number of timesteps to evolve the system by each time the "
        "sterics interactions are modified. A value of 1 will give a 'smooth' "
        "transition between the each discrete lambda value whereas a value greater "
        "than 1 will yield a stepwise transition as shown in Figure 3 of "
        "doi:10.1063/1.4712028.",
    )


class NonEquilibriumProtocol(pydantic.BaseModel):
    """Configure a non-equilibrium alchemical free energy calculation [1, 2].

    It is expected that first the electrostatics interactions will be annihilated
    followed by a decoupling of the sterics interactions.

    References:
        [1] Ballard, Andrew J., and Christopher Jarzynski. "Replica exchange with
            nonequilibrium switches: Enhancing equilibrium sampling by increasing
            replica overlap." The Journal of chemical physics 136.19 (2012): 194101.

        [2] Gapsys, Vytautas, et al. "Large scale relative protein ligand binding
            affinities using non-equilibrium alchemy." Chemical Science 11.4 (2020):
            1140-1152.
    """

    type: typing.Literal["non-equilibrium"] = "non-equilibrium"

    minimization_protocol: MinimizationProtocol | None = pydantic.Field(
        MinimizationProtocol(),
        description="The (optional) protocol to follow when minimizing the system in "
        "both the end states prior to running the equilibrium simulations.",
    )
    equilibration_protocol: SimulationProtocol | None = pydantic.Field(
        SimulationProtocol(n_steps=100000),  # 200 ps
        description="The (optional) protocol to follow when equilibrating the system "
        "in both the end states prior to running the production equilibrium "
        "simulations.",
    )

    production_protocol: SimulationProtocol = pydantic.Field(
        SimulationProtocol(n_steps=6250 * 160),  # 2 ns
        description="The protocol to follow when running the production equilibrium "
        "simulation in both the end states. The snapshots generated at the end of each "
        "iteration will be used for each non-equilibrium switch.",
    )
    production_report_interval: pydantic.PositiveInt = pydantic.Field(
        6250,
        description="The interval at which to write out the simulation state during "
        "the production simulation.",
    )

    switching_protocol: SwitchingProtocol = pydantic.Field(
        ...,
        description="The protocol that describes how each snapshot generated during "
        "the production simulation should be driven from state 0 -> 1 and likewise "
        "1 -> 0 in order to compute the non-equilibrium work distributions.",
    )


AlchemicalProtocol = EquilibriumProtocol | NonEquilibriumProtocol


class Config(pydantic.BaseModel):
    """A schema that fully defines the inputs needed to compute the transfer free energy
    of a solvent between to solvents, or between a solvent and vacuum."""

    system: System = pydantic.Field(
        ...,
        description="A description of the system under investigation, including the "
        "identity of the solute and the two solvent phases.",
    )

    temperature: OpenMMQuantity[openmm.unit.kelvin] = pydantic.Field(
        ..., description="The temperature to calculate at [K]."
    )
    pressure: OpenMMQuantity[openmm.unit.atmosphere] | None = pydantic.Field(
        ..., description="The pressure to calculate at [atm]."
    )

    alchemical_protocol_a: AlchemicalProtocol = pydantic.Field(
        ...,
        description="The protocol that describes the alchemical pathway to transform "
        "the solute along in the first solvent.",
    )
    alchemical_protocol_b: AlchemicalProtocol = pydantic.Field(
        ...,
        description="The protocol that describes the alchemical pathway to transform "
        "the solute along in the second solvent.",
    )


class Result(pydantic.BaseModel):
    """The result of a free energy calculation."""

    dg_solvent_a: OpenMMQuantity[KCAL_MOL] = pydantic.Field(
        description="The change in free energy of alchemically transforming the "
        "solute from an interacting to a non-interacting state in the first solvent.",
    )
    dg_std_solvent_a: OpenMMQuantity[KCAL_MOL] = pydantic.Field(
        description="The standard error in ``dg_solvent_a``."
    )

    dg_solvent_b: OpenMMQuantity[KCAL_MOL] = pydantic.Field(
        description="The change in free energy of alchemically transforming the "
        "solute from an interacting to a non-interacting state in the second solvent.",
    )
    dg_std_solvent_b: OpenMMQuantity[KCAL_MOL] = pydantic.Field(
        description="The standard error in ``dg_solvent_b``."
    )

    @property
    def dg(self) -> openmm.unit.Quantity:
        """The change in free energy of transferring the solute from *solvent-a* to
        *solvent-b* in units of kT."""
        return self.dg_solvent_b - self.dg_solvent_a

    @property
    def dg_std(self) -> openmm.unit.Quantity:
        """The standard error in ``ddg``."""

        std = math.sqrt(
            self.dg_std_solvent_a.value_in_unit(KCAL_MOL) ** 2
            + self.dg_std_solvent_b.value_in_unit(KCAL_MOL) ** 2
        )
        return std * KCAL_MOL

    def __str__(self):
        return (
            f"ΔG a->b={self.dg.value_in_unit(KCAL_MOL):.3f} kcal/mol "
            f"ΔG a->b std={self.dg_std.value_in_unit(KCAL_MOL):.3f} kcal/mol"
        )

    def __repr__(self):
        return f"{self.__repr_name__()}({self.__str__()})"
