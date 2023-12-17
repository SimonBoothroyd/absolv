"""Run non-equilibrium forward and reverse sampling."""
import typing

import numpy
import openmm.app
import openmm.unit
import tqdm

import absolv.config
import absolv.fep
import absolv.utils.openmm


def _compute_lambdas(
    protocol: absolv.config.SwitchingProtocol,
    step: int,
    timestep: openmm.unit.Quantity,
    reverse_direction: bool,
) -> tuple[float, float, float]:
    """Computes the values of the global, electrostatic and steric lambdas parameters
    at a given step.

    Args:
        protocol: The switching protocol that defines the lambda schedule.
        step: The current step number.
        timestep: The integrator timestep being used.
        reverse_direction: Whether to move from state 1 -> 0 rather than 0 -> 1.

    Returns:
        The values of the global, electrostatic and steric lambdas parameters.
    """

    n_electrostatic_steps = (
        protocol.n_electrostatic_steps * protocol.n_steps_per_electrostatic_step
    )
    n_steric_steps = protocol.n_steric_steps * protocol.n_steps_per_steric_step

    n_total_steps = n_electrostatic_steps + n_steric_steps

    if reverse_direction:
        step = n_total_steps - step

    time = step * timestep

    time_electrostatics = timestep * n_electrostatic_steps
    time_total = timestep * n_total_steps

    lambda_global = (time_total - time) / time_total

    lambda_electrostatics = (
        0.0
        if step >= n_electrostatic_steps or n_electrostatic_steps == 0
        else (
            (time_electrostatics + time_total * (lambda_global - 1.0))
            / time_electrostatics
        )
    )
    lambda_sterics = (
        1.0
        if step <= n_electrostatic_steps
        else (time_total / (time_total - time_electrostatics) * lambda_global)
    )

    return lambda_global, lambda_electrostatics, lambda_sterics


def _enumerate_frames(
    protocol: absolv.config.SwitchingProtocol, reverse_direction: bool
) -> typing.Iterable[tuple[int, int]]:
    """An iterator that enumerates all frame indices."""

    stages = (
        (
            protocol.n_electrostatic_steps + (0 if not reverse_direction else 1),
            protocol.n_steps_per_electrostatic_step,
        ),
        (
            protocol.n_steric_steps + (0 if reverse_direction else 1),
            protocol.n_steps_per_steric_step,
        ),
    )

    frame_idx = 0

    for i, (n_lambda_steps, n_steps_per_lambda) in enumerate(
        stages if not reverse_direction else reversed(stages)
    ):
        for _ in tqdm.tqdm(range(n_lambda_steps - int(i + 1 == len(stages)))):
            yield frame_idx, n_steps_per_lambda
            frame_idx += n_steps_per_lambda


def _compute_reduced_potential(
    simulation: openmm.app.Simulation,
    temperature: openmm.unit.Quantity,
    pressure: openmm.unit.Quantity | None,
) -> float:
    """Computes the reduced potential of the contexts current state.

    Returns:
        The reduced potential.
    """

    state = simulation.context.getState(getEnergy=True)

    unreduced_potential = state.getPotentialEnergy() / openmm.unit.AVOGADRO_CONSTANT_NA

    if pressure is not None:
        unreduced_potential += pressure * state.getPeriodicBoxVolume()

    beta = 1.0 / (openmm.unit.BOLTZMANN_CONSTANT_kB * temperature)
    return unreduced_potential * beta


def _simulate(
    simulation: openmm.app.Simulation,
    coords: openmm.State,
    protocol: absolv.config.SwitchingProtocol,
    reverse_direction: bool,
) -> numpy.ndarray:
    """Evolve the state of the context according to a specific protocol."""

    timestep = simulation.integrator.getStepSize()

    barostats = [
        force
        for force in simulation.context.getSystem().getForces()
        if isinstance(force, openmm.MonteCarloBarostat)
    ]
    assert len(barostats) <= 1, "a maximum of one barostat is supported."

    pressure = None if len(barostats) == 0 else barostats[0].getDefaultPressure()
    temperature = simulation.integrator.getTemperature()

    _, lambda_electrostatics, lambda_sterics = _compute_lambdas(
        protocol, 0, timestep, reverse_direction
    )

    simulation.context.setState(coords)

    absolv.fep.set_fep_lambdas(
        simulation.context, lambda_sterics, lambda_electrostatics
    )

    reduced_potentials = []

    for step, n_steps_per_lambda in _enumerate_frames(protocol, reverse_direction):
        u_old = _compute_reduced_potential(simulation, temperature, pressure)

        (_, lambda_electrostatics, lambda_sterics) = _compute_lambdas(
            protocol, step + n_steps_per_lambda, timestep, reverse_direction
        )
        absolv.fep.set_fep_lambdas(
            simulation.context, lambda_sterics, lambda_electrostatics
        )

        u_new = _compute_reduced_potential(simulation, temperature, pressure)
        reduced_potentials.append((u_old, u_new))

        simulation.step(n_steps_per_lambda)

    return numpy.array(reduced_potentials)


def run_neq(
    simulation: openmm.app.Simulation,
    coords_0: openmm.State,
    coords_1: openmm.State,
    protocol: absolv.config.SwitchingProtocol,
) -> tuple[float, float]:
    """Run a non-equilibrium simulation with OpenMM, whereby a system is non-reversibly
    pulled along an alchemical pathway as described by Ballard and Jarzynski [1]
    (Figure 3) and Gapsys et al [2].

    Both the forward and reverse directions will be simulated.

    References:
        [1] Ballard, Andrew J., and Christopher Jarzynski. "Replica exchange with
        nonequilibrium switches: Enhancing equilibrium sampling by increasing replica
        overlap." The Journal of chemical physics 136.19 (2012): 194101.

        [2] Gapsys, Vytautas, et al. "Large scale relative protein ligand binding
        affinities using non-equilibrium alchemy." Chemical Science 11.4 (2020):
        1140-1152.

    Returns:
        The forward and reverse work values.
    """

    forward_potentials = _simulate(
        simulation, coords_0, protocol, reverse_direction=False
    )
    reverse_potentials = _simulate(
        simulation, coords_1, protocol, reverse_direction=True
    )

    forward_work = (forward_potentials[:, 1] - forward_potentials[:, 0]).sum()
    reverse_work = (reverse_potentials[:, 1] - reverse_potentials[:, 0]).sum()

    return forward_work, reverse_work
