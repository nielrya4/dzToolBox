"""
dz_lib - Detrital Zircon Analysis Library

A Python library for detrital zircon geochronology analysis.

Modules:
    - univariate: Single-variable analysis (age distributions, MDA, etc.)
    - bivariate: Two-variable analysis
    - utils: Utility functions

Configuration:
    - config: Global settings including sigma level preferences
"""

from . import config
from . import univariate
from . import bivariate
from . import utils

__version__ = "2.0.0"

__all__ = [
    "config",
    "univariate",
    "bivariate",
    "utils",
]
