import functools
import typing

import numpy
import openmm.unit


def _compare_values(
    a: float | openmm.unit.Quantity,
    b: float | openmm.unit.Quantity,
    predicate: typing.Callable[[float | numpy.ndarray, float | numpy.ndarray], bool],
) -> bool:
    """Compare to values using a specified predicate taking units into account."""

    if isinstance(a, openmm.unit.Quantity):
        expected_unit = a.unit
        a = a.value_in_unit(expected_unit)

    else:
        expected_unit = None

    if isinstance(b, openmm.unit.Quantity):
        assert expected_unit is not None, "cannot compare quantity with unit-less."
        b = b.value_in_unit(expected_unit)

    else:
        assert expected_unit is None, "cannot compare quantity with unit-less."

    return predicate(a, b)


def is_close(
    a: float | openmm.unit.Quantity,
    b: float | openmm.unit.Quantity,
    rtol=1.0e-5,
    atol=1.0e-8,
) -> bool:
    """Compare whether two values are close taking units into account."""

    return _compare_values(a, b, functools.partial(numpy.isclose, rtol=rtol, atol=atol))


def all_close(
    a: numpy.ndarray | openmm.unit.Quantity,
    b: numpy.ndarray | openmm.unit.Quantity,
    rtol=1.0e-5,
    atol=1.0e-8,
) -> bool:
    """Compare whether all elements in two array are close taking units into account."""

    return _compare_values(
        a, b, functools.partial(numpy.allclose, rtol=rtol, atol=atol)
    )
