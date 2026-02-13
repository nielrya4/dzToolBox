"""
Multivariate sample and grain data structures for tensor factorization.

These classes handle grain-level data with multiple features (e.g., Age, Eu_anomaly, Ti_temp, etc.)
as opposed to the simple univariate (age, uncertainty) format.
"""

import numpy as np
from typing import Dict, List, Optional


class MultivariateGrain:
    """
    Represents a single grain with multiple feature measurements.

    Attributes:
        grain_id: Unique identifier for the grain
        features: Dictionary mapping feature names to values
            Example: {'Age': 116.74, 'Eu_anomaly': 0.242751, 'Ti_temp': 617.85, ...}
    """

    def __init__(self, grain_id: str, features: Dict[str, float]):
        """
        Initialize a multivariate grain.

        Parameters:
            grain_id: Unique identifier (e.g., 'GRAIN ID 1')
            features: Dict of feature_name -> value mappings
        """
        self.grain_id = str(grain_id)
        self.features = features

    def get_feature_vector(self, feature_names: List[str]) -> np.ndarray:
        """
        Get feature values as a numpy array in specified order.

        Parameters:
            feature_names: Ordered list of feature names

        Returns:
            1D numpy array of feature values

        Raises:
            KeyError: If a requested feature is not in this grain
        """
        return np.array([self.features[name] for name in feature_names])

    def to_dict(self) -> dict:
        """Convert grain to dictionary representation."""
        return {
            'grain_id': self.grain_id,
            'features': self.features
        }

    def __repr__(self):
        feature_str = ', '.join(f"{k}={v:.2f}" for k, v in list(self.features.items())[:3])
        if len(self.features) > 3:
            feature_str += ', ...'
        return f"MultivariateGrain(id='{self.grain_id}', {feature_str})"


class MultivariateSample:
    """
    Collection of grains from the same sample (SINK_ID).

    Attributes:
        name: Sample name (SINK_ID)
        grains: List of MultivariateGrain objects
        feature_names: Ordered list of feature names
    """

    def __init__(self, name: str, grains: List[MultivariateGrain], feature_names: List[str]):
        """
        Initialize a multivariate sample.

        Parameters:
            name: Sample name (e.g., 'SINK NAME 1')
            grains: List of MultivariateGrain objects
            feature_names: Ordered list of feature names (must match grain features)

        Raises:
            ValueError: If grains have inconsistent features or empty grain list
        """
        self.name = str(name)
        self.grains = grains
        self.feature_names = feature_names

        if not grains:
            raise ValueError(f"Sample '{name}' has no grains")

        # Validate that all grains have the same features
        for grain in grains:
            if set(grain.features.keys()) != set(feature_names):
                missing = set(feature_names) - set(grain.features.keys())
                extra = set(grain.features.keys()) - set(feature_names)
                raise ValueError(
                    f"Grain '{grain.grain_id}' in sample '{name}' has inconsistent features. "
                    f"Missing: {missing}, Extra: {extra}"
                )

    def get_feature_matrix(self) -> np.ndarray:
        """
        Get all grains' features as a 2D matrix.

        Returns:
            2D numpy array of shape (n_grains, n_features)
            Each row is one grain, columns are features in self.feature_names order
        """
        return np.array([grain.get_feature_vector(self.feature_names) for grain in self.grains])

    def get_grain_count(self) -> int:
        """Get the number of grains in this sample."""
        return len(self.grains)

    def to_dict(self) -> dict:
        """Convert sample to dictionary representation."""
        return {
            'name': self.name,
            'feature_names': self.feature_names,
            'grain_count': self.get_grain_count(),
            'grains': [grain.to_dict() for grain in self.grains]
        }

    def __repr__(self):
        return (f"MultivariateSample(name='{self.name}', "
                f"n_grains={self.get_grain_count()}, "
                f"n_features={len(self.feature_names)})")
