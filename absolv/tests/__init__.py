import functools
from typing import Callable, Union

import numpy
from openmm import unit


def _compare_values(
    a: Union[float, unit.Quantity],
    b: Union[float, unit.Quantity],
    predicate: Callable[
        [Union[float, numpy.ndarray], Union[float, numpy.ndarray]], bool
    ],
) -> bool:
    """Compare to values using a specified predicate taking units into account."""

    if isinstance(a, unit.Quantity):

        expected_unit = a.unit
        a = a.value_in_unit(expected_unit)

    else:

        expected_unit = None

    if isinstance(b, unit.Quantity):

        assert expected_unit is not None, "cannot compare quantity with unit-less."
        b = b.value_in_unit(expected_unit)

    else:

        assert expected_unit is None, "cannot compare quantity with unit-less."

    return predicate(a, b)


def is_close(
    a: Union[float, unit.Quantity],
    b: Union[float, unit.Quantity],
    rtol=1.0e-5,
    atol=1.0e-8,
) -> bool:
    """Compare whether two values are close taking units into account."""

    return _compare_values(a, b, functools.partial(numpy.isclose, rtol=rtol, atol=atol))


def all_close(
    a: Union[numpy.ndarray, unit.Quantity],
    b: Union[numpy.ndarray, unit.Quantity],
    rtol=1.0e-5,
    atol=1.0e-8,
) -> bool:
    """Compare whether all elements in two array are close taking units into account."""

    return _compare_values(
        a, b, functools.partial(numpy.allclose, rtol=rtol, atol=atol)
    )
