"""
dz_lib Configuration Module

Provides global settings for the dz_lib library, including sigma level preferences.

Sigma Levels:
    - sigma_in: The sigma level of input uncertainties (typically 1 or 2)
    - sigma_out: The sigma level for output uncertainties (0, 1, 2, etc.)

Usage:
    from dz_lib import config

    # Set global defaults
    config.set_sigma_in(1)      # Input uncertainties are 1-sigma
    config.set_sigma_out(2)     # Output uncertainties at 2-sigma

    # Get current settings
    config.get_sigma_in()       # Returns 1
    config.get_sigma_out()      # Returns 2

    # Convert uncertainties
    config.to_sigma(unc_1s, from_sigma=1, to_sigma=2)  # Returns unc_2s
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DZConfig:
    """Global configuration for dz_lib."""

    # Sigma level of input uncertainties (1 or 2)
    sigma_in: int = 1

    # Sigma level for output uncertainties (0, 1, 2, ...)
    sigma_out: int = 2

    def validate(self):
        """Validate configuration values."""
        if self.sigma_in not in (1, 2):
            raise ValueError(f"sigma_in must be 1 or 2, got {self.sigma_in}")
        if self.sigma_out < 0:
            raise ValueError(f"sigma_out must be non-negative, got {self.sigma_out}")


# Global configuration instance
_config = DZConfig()


def get_config() -> DZConfig:
    """Get the current configuration."""
    return _config


def set_sigma_in(sigma: int) -> None:
    """Set the sigma level of input uncertainties (1 or 2)."""
    if sigma not in (1, 2):
        raise ValueError(f"sigma_in must be 1 or 2, got {sigma}")
    _config.sigma_in = sigma


def get_sigma_in() -> int:
    """Get the sigma level of input uncertainties."""
    return _config.sigma_in


def set_sigma_out(sigma: int) -> None:
    """Set the sigma level for output uncertainties (0, 1, 2, ...)."""
    if sigma < 0:
        raise ValueError(f"sigma_out must be non-negative, got {sigma}")
    _config.sigma_out = sigma


def get_sigma_out() -> int:
    """Get the sigma level for output uncertainties."""
    return _config.sigma_out


def to_sigma(
    uncertainty: float | np.ndarray,
    from_sigma: int = 1,
    to_sigma: Optional[int] = None
) -> float | np.ndarray:
    """
    Convert uncertainty from one sigma level to another.

    Parameters
    ----------
    uncertainty : float or array
        The uncertainty value(s) to convert
    from_sigma : int
        The sigma level the uncertainty is currently at (default: 1)
    to_sigma : int, optional
        The sigma level to convert to (default: uses config.sigma_out)

    Returns
    -------
    float or array
        The converted uncertainty value(s)

    Examples
    --------
    >>> to_sigma(5.0, from_sigma=1, to_sigma=2)
    10.0
    >>> to_sigma(10.0, from_sigma=2, to_sigma=1)
    5.0
    """
    if to_sigma is None:
        to_sigma = _config.sigma_out

    if from_sigma == 0:
        return 0.0 if isinstance(uncertainty, (int, float)) else np.zeros_like(uncertainty)

    if to_sigma == 0:
        return 0.0 if isinstance(uncertainty, (int, float)) else np.zeros_like(uncertainty)

    # Convert to 1-sigma first, then to target
    unc_1s = uncertainty / from_sigma
    return unc_1s * to_sigma


def normalize_to_1s(
    uncertainty: float | np.ndarray,
    sigma_in: Optional[int] = None
) -> float | np.ndarray:
    """
    Normalize uncertainty to 1-sigma.

    Parameters
    ----------
    uncertainty : float or array
        The uncertainty value(s)
    sigma_in : int, optional
        The sigma level of the input (default: uses config.sigma_in)

    Returns
    -------
    float or array
        The uncertainty at 1-sigma
    """
    if sigma_in is None:
        sigma_in = _config.sigma_in
    return to_sigma(uncertainty, from_sigma=sigma_in, to_sigma=1)


def format_uncertainty(
    age: float,
    uncertainty_1s: float,
    sigma: Optional[int] = None,
    precision: int = 2
) -> str:
    """
    Format an age with uncertainty at the specified sigma level.

    Parameters
    ----------
    age : float
        The age value
    uncertainty_1s : float
        The 1-sigma uncertainty
    sigma : int, optional
        The sigma level to report (default: uses config.sigma_out)
    precision : int
        Number of decimal places

    Returns
    -------
    str
        Formatted string like "100.5 +/- 2.3 Ma (2s)"
    """
    if sigma is None:
        sigma = _config.sigma_out

    unc = uncertainty_1s * sigma

    if sigma == 0:
        return f"{age:.{precision}f} Ma"
    elif sigma == 1:
        return f"{age:.{precision}f} +/- {unc:.{precision}f} Ma (1s)"
    elif sigma == 2:
        return f"{age:.{precision}f} +/- {unc:.{precision}f} Ma (2s)"
    else:
        return f"{age:.{precision}f} +/- {unc:.{precision}f} Ma ({sigma}s)"
