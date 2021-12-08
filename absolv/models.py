from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy
from openmm import unit
from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    confloat,
    root_validator,
    validator,
)


def float_validator(field_name: str, expected_units: unit.Unit) -> validator:
    def validate_unit(cls, value):

        if isinstance(value, str):
            return float(value)
        elif value is None or isinstance(value, float):
            return value

        assert isinstance(value, unit.Quantity)
        return value.value_in_unit(expected_units)

    return validator(field_name, allow_reuse=True, pre=True)(validate_unit)


class System(BaseModel):
    """A model that describes the contents of the two solvents that the solutes will
    be transferred between (a -> b) as well as the solutes themselves.
    """

    solutes: Dict[str, PositiveInt] = Field(
        ...,
        description="A dictionary containing the SMILES patterns of each solute in the "
        "system as well as how many instances of each there should be.",
    )

    solvent_a: Optional[Dict[str, PositiveInt]] = Field(
        ...,
        description="A dictionary containing the SMILES patterns of each component "
        "in the first solvent as well as how many instances of each there should be."
        "A value of ``None`` should be used to indicate vacuum.",
    )
    solvent_b: Optional[Dict[str, PositiveInt]] = Field(
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

    @validator("solutes")
    def _validate_solutes(cls, value):

        assert len(value) > 0, "at least one solute must be specified"
        return value

    @validator("solvent_a")
    def _validate_solvent_a(cls, value):

        if value is None:
            return value

        assert (
            len(value) > 0
        ), "at least one solvent must be specified when `solvent_a` is not none"
        return value

    @validator("solvent_b")
    def _validate_solvent_b(cls, value):

        if value is None:
            return value

        assert (
            len(value) > 0
        ), "at least one solvent must be specified when `solvent_b` is not none"
        return value

    def to_components(self) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """Converts this object into two lists: one containing the identities and amounts
        of the molecules present in the first solution, and one the containg the same for
        the second solution.

        The identity and amount is stored in a tuple as a SMILES pattern and integer
        count.
        """

        components_a = [*self.solutes.items()] + (
            [] if self.solvent_a is None else [*self.solvent_a.items()]
        )
        components_b = [*self.solutes.items()] + (
            [] if self.solvent_b is None else [*self.solvent_b.items()]
        )

        return components_a, components_b


class State(BaseModel):
    """A model that defines the temperature and (optionally) the pressure of
    a thermodynamic state."""

    temperature: PositiveFloat = Field(..., description="The temperature [K].")
    pressure: Optional[PositiveFloat] = Field(..., description="The pressure [atm].")

    _validate_temperature = float_validator("temperature", unit.kelvin)
    _validate_pressure = float_validator("pressure", unit.atmosphere)


class MinimizationProtocol(BaseModel):
    """A model that encodes how a system should be energy minimized."""

    tolerance: PositiveFloat = Field(
        10.0,
        description="How precisely the energy minimum must be located [kj / mol / nm]",
    )
    max_iterations: NonNegativeInt = Field(
        0,
        description="The maximum number of iterations to perform. If this is 0, "
        "minimization is continued until the results converge.",
    )

    _validate_tolerance = float_validator(
        "tolerance", unit.kilojoule_per_mole / unit.nanometers
    )


class SimulationProtocol(BaseModel):
    """A model that encodes how a system should be evolved by molecular simulation."""

    n_steps_per_iteration: PositiveInt = Field(
        ...,
        description="The number of steps to evolve the system by at each iteration. "
        "The total number of steps performed will be equal to the "
        "``total_number_of_iterations * steps_per_iteration``.",
    )
    n_iterations: PositiveInt = Field(
        ...,
        description="The number of times to evolve the system forward by "
        "``n_steps_per_iteration`` time steps.",
    )

    timestep: PositiveFloat = Field(
        2.0, description="The timestep [fs] to evolve the system by at each step."
    )
    thermostat_friction: PositiveFloat = Field(
        1.0,
        description="The friction coefficient [1/ps] to use for the Langevin "
        "thermostat.",
    )

    _validate_timestep = float_validator("timestep", unit.femtoseconds)
    _validate_thermostat_friction = float_validator(
        "thermostat_friction", (unit.picoseconds ** -1)
    )


class EquilibriumProtocol(BaseModel):
    """A model that encodes the protocol for performing an equilibrium (e.g. TI, MBAR)
    alchemical free energy calculation.
    """

    type: Literal["equilibrium"] = "equilibrium"

    minimization_protocol: Optional[MinimizationProtocol] = Field(
        MinimizationProtocol(),
        description="Whether to minimize the energy of the system prior to any "
        "simulations.",
    )

    equilibration_protocol: Optional[SimulationProtocol] = Field(
        SimulationProtocol(n_steps_per_iteration=10000, n_iterations=10),  # 200 ps
        description="The (optional) protocol that describes the equilibration "
        "simulation to run prior to the production one.",
    )
    production_protocol: SimulationProtocol = Field(
        SimulationProtocol(n_steps_per_iteration=6250, n_iterations=160),  # 2 ns
        description="The protocol that describes the production to run.",
    )

    lambda_sterics: List[confloat(ge=0.0, le=1.0)] = Field(
        ...,
        description="The alchemical pathway to transform the vdW interactions along. A "
        "value of 1.0 represents a fully interacting system while a value of 0.0 "
        "represents a system with the solute-solute and solute-solvent vdW interactions "
        "disabled.",
    )
    lambda_electrostatics: List[confloat(ge=0.0, le=1.0)] = Field(
        ...,
        description="The alchemical pathway to transform the electrostatic interactions "
        "along. A value of 1.0 represents a fully interacting system while a value of "
        "0.0 represents a system with the solute-solute and solute-solvent "
        "electrostatic interactions disabled.",
    )

    @property
    def n_states(self) -> int:
        """Returns the number of lambda states that will be sampled at."""
        return len(self.lambda_sterics)

    @root_validator
    def _validate_lambda_lengths(cls, values):

        lambda_names = ["lambda_sterics", "lambda_electrostatics"]
        lambda_lists = [values.get(lambda_name) for lambda_name in lambda_names]

        assert all(
            len(lambda_list) == len(lambda_lists[0]) for lambda_list in lambda_lists
        ), "lambda lists must be the same length"

        return values


class SwitchingProtocol(BaseModel):
    """A model that encodes the protocol for non-reversibly driving a system between
    to alchemical states."""

    n_electrostatic_steps: NonNegativeInt = Field(
        ...,
        description="The number of steps to annihilate the electrostatics interactions "
        "over. The total time needed to annihilate the electrostatics interactions will "
        "be ``n_electrostatic_steps * n_steps_per_electrostatic_step * timestep``",
    )
    n_steps_per_electrostatic_step: NonNegativeInt = Field(
        ...,
        description="The number of timesteps to evolve the system by each time the "
        "electrostatics interactions are modified. A value of 1 will give a 'smooth' "
        "transition between the each discrete lambda value whereas a value greater than "
        "1 will yield a stepwise transition as shown in Figure 3 of "
        "doi:10.1063/1.4712028.",
    )

    n_steric_steps: NonNegativeInt = Field(
        ...,
        description="The number of steps to decouple the sterics interactions over once "
        "the electrostatics interactions have been annihilated. The total time needed "
        "to annihilate the sterics interactions will be "
        "``n_steric_steps * n_steps_per_steric_step * timestep``",
    )
    n_steps_per_steric_step: NonNegativeInt = Field(
        ...,
        description="The number of timesteps to evolve the system by each time the "
        "sterics interactions are modified. A value of 1 will give a 'smooth' "
        "transition between the each discrete lambda value whereas a value greater than "
        "1 will yield a stepwise transition as shown in Figure 3 of "
        "doi:10.1063/1.4712028.",
    )

    timestep: PositiveFloat = Field(
        2.0, description="The timestep [fs] to evolve the system by at each step."
    )
    thermostat_friction: PositiveFloat = Field(
        1.0,
        description="The friction coefficient [1/ps] to use for the Langevin "
        "thermostat.",
    )

    _validate_timestep = float_validator("timestep", unit.femtoseconds)
    _validate_thermostat_friction = float_validator(
        "thermostat_friction", (unit.picoseconds ** -1)
    )


class NonEquilibriumProtocol(BaseModel):
    """A model that encodes the protocol for performing a non-equilibrium switching
    like alchemical free energy calculation [1, 2].

    It is expected that first the electrostatics interactions will be annihilated
    followed by a decoupling of the sterics interactions.

    References:
        [1] Ballard, Andrew J., and Christopher Jarzynski. "Replica exchange with
            nonequilibrium switches: Enhancing equilibrium sampling by increasing replica
            overlap." The Journal of chemical physics 136.19 (2012): 194101.

        [2] Gapsys, Vytautas, et al. "Large scale relative protein ligand binding
            affinities using non-equilibrium alchemy." Chemical Science 11.4 (2020):
            1140-1152.
    """

    type: Literal["non-equilibrium"] = "non-equilibrium"

    minimization_protocol: Optional[MinimizationProtocol] = Field(
        MinimizationProtocol(),
        description="The (optional) protocol to follow when minimizing the system in "
        "both the end states prior to running the equilibrium simulations.",
    )

    equilibration_protocol: Optional[SimulationProtocol] = Field(
        SimulationProtocol(n_steps_per_iteration=10000, n_iterations=10),  # 200 ps
        description="The (optional) protocol to follow when equilibrating the system in "
        "both the end states prior to running the production equilibrium simulations.",
    )
    production_protocol: SimulationProtocol = Field(
        SimulationProtocol(n_steps_per_iteration=6250, n_iterations=160),  # 2 ns
        description="The protocol to follow when running the production equilibrium "
        "simulation in both the end states. The snapshots generated at the end of each "
        "iteration will be used for each non-equilibrium switch.",
    )

    switching_protocol: SwitchingProtocol = Field(
        ...,
        description="The protocol that describes how each snapshot generated during the "
        "production simulation should be driven from state 0 -> 1 and likewise 1 -> 0 "
        "in order to compute the non-equilibrium work distributions.",
    )


class TransferFreeEnergySchema(BaseModel):
    """A schema that fully defines the inputs needed to compute the transfer free energy
    of a solvent between to solvents, or between a solvent and vacuum."""

    system: System = Field(
        ...,
        description="A description of the system under investigation, including the "
        "identity of the solute and the two solvent phases.",
    )
    state: State = Field(
        ..., description="The thermodynamic state to perform the calculation at."
    )

    alchemical_protocol_a: Union[EquilibriumProtocol, NonEquilibriumProtocol] = Field(
        ...,
        description="The protocol that describes the alchemical pathway to transform "
        "the solute along in the first solvent.",
    )
    alchemical_protocol_b: Union[EquilibriumProtocol, NonEquilibriumProtocol] = Field(
        ...,
        description="The protocol that describes the alchemical pathway to transform "
        "the solute along in the second solvent.",
    )


class DeltaG(BaseModel):

    value: float = Field(..., description="The value of the free energy in units of kT")
    std_error: float = Field(
        ..., description="The standard error of the value in units of kT"
    )

    def __add__(self, other: "DeltaG") -> "DeltaG":

        if not isinstance(other, DeltaG):
            raise NotImplementedError

        return DeltaG(
            value=self.value + other.value,
            std_error=numpy.sqrt(self.std_error ** 2 + other.std_error ** 2),
        )

    def __sub__(self, other: "DeltaG") -> "DeltaG":

        if not isinstance(other, DeltaG):
            raise NotImplementedError

        return DeltaG(
            value=self.value - other.value,
            std_error=numpy.sqrt(self.std_error ** 2 + other.std_error ** 2),
        )


class TransferFreeEnergyResult(BaseModel):

    input_schema: TransferFreeEnergySchema = Field(
        ..., description="The schema that was used to generate this result."
    )

    delta_g_solvent_a: DeltaG = Field(
        ...,
        description="The change in free energy of alchemically transforming the solute "
        "from an interacting to a non-interacting state in *solvent-a*.",
    )
    delta_g_solvent_b: DeltaG = Field(
        ...,
        description="The change in free energy of alchemically transforming the solute "
        "from an interacting to a non-interacting state in *solvent-b*.",
    )

    provenance: Dict[str, Any] = Field(
        {}, description="Extra provenance about how this result was generated."
    )

    @property
    def delta_g_from_a_to_b(self) -> DeltaG:
        """The change in free energy of transferring the solute from *solvent-a* to
        *solvent-b* in units of kT."""
        return self.delta_g_solvent_a - self.delta_g_solvent_b

    @property
    def delta_g_from_b_to_a(self) -> DeltaG:
        """The change in free energy of transferring the solute from *solvent-b* to
        *solvent-a* in units of kT."""
        return self.delta_g_solvent_b - self.delta_g_solvent_a

    @property
    def _boltzmann_temperature(self) -> unit.Quantity:
        return (
            unit.MOLAR_GAS_CONSTANT_R
            * unit.kelvin
            * self.input_schema.state.temperature
        ).in_units_of(unit.kilocalories_per_mole)

    @property
    def delta_g_from_a_to_b_with_units(self) -> Tuple[unit.Quantity, unit.Quantity]:
        """The change in free energy of transferring the solute from *solvent-a* to
        *solvent-b*, as well as the error in that change."""

        return (
            self.delta_g_from_a_to_b.value * self._boltzmann_temperature,
            self.delta_g_from_a_to_b.std_error * self._boltzmann_temperature,
        )

    @property
    def delta_g_from_b_to_a_with_units(self) -> Tuple[unit.Quantity, unit.Quantity]:
        """The change in free energy of transferring the solute from *solvent-a* to
        *solvent-b*, as well as the error in that change."""

        return (
            self.delta_g_from_b_to_a.value * self._boltzmann_temperature,
            self.delta_g_from_b_to_a.std_error * self._boltzmann_temperature,
        )

    def __str__(self):

        value_a_b, std_error_a_b = self.delta_g_from_a_to_b_with_units

        return (
            f"ΔG a->b={value_a_b.value_in_unit(unit.kilocalories_per_mole):.3f} kcal/mol "
            f"ΔG a->b std={std_error_a_b.value_in_unit(unit.kilocalories_per_mole):.3f} kcal/mol"
        )

    def __repr__(self):
        return f"{self.__repr_name__()}({self.__str__()})"
