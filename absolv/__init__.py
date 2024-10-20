"""Absolute solvation free energy calculations using OpenMM"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("absolv")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
