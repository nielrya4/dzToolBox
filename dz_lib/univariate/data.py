import numpy as np
from typing import Optional, List, Union


class Grain:
    """
    Represents a single detrital grain with age and uncertainty.

    Parameters
    ----------
    age : float
        The age of the grain in Ma
    uncertainty : float
        The uncertainty of the age measurement
    sigma_in : int, optional
        The sigma level of the uncertainty (1 or 2). If not specified,
        uses the global config.sigma_in setting.
    """
    def __init__(self, age: float, uncertainty: float, sigma_in: Optional[int] = None):
        from dz_lib import config

        self.age = age
        self._uncertainty_raw = uncertainty
        self._sigma_in = sigma_in if sigma_in is not None else config.get_sigma_in()

    @property
    def uncertainty(self) -> float:
        """Returns uncertainty at the input sigma level."""
        return self._uncertainty_raw

    @property
    def uncertainty_1s(self) -> float:
        """Returns uncertainty normalized to 1-sigma."""
        if self._sigma_in == 1:
            return self._uncertainty_raw
        return self._uncertainty_raw / self._sigma_in

    @property
    def uncertainty_2s(self) -> float:
        """Returns uncertainty at 2-sigma."""
        return self.uncertainty_1s * 2

    def get_uncertainty(self, sigma: Optional[int] = None) -> float:
        """
        Get uncertainty at specified sigma level.

        Parameters
        ----------
        sigma : int, optional
            Target sigma level. If None, uses config.sigma_out.
        """
        from dz_lib import config
        if sigma is None:
            sigma = config.get_sigma_out()
        return self.uncertainty_1s * sigma

    def to_dict(self):
        return {
            'age': self.age,
            'uncertainty': self.uncertainty,
            'sigma_in': self._sigma_in
        }


class Sample:
    """
    Represents a collection of detrital grains from a sample.

    Parameters
    ----------
    name : str
        Sample name/identifier
    grains : list of Grain
        The grains in this sample
    sigma_in : int, optional
        Default sigma level for grains. Individual grains may override this.
    """
    def __init__(self, name: str, grains: List[Grain], sigma_in: Optional[int] = None):
        from dz_lib import config

        self.name = name
        self.grains = grains
        self._sigma_in = sigma_in if sigma_in is not None else config.get_sigma_in()

    @property
    def sigma_in(self) -> int:
        """The sigma level of input uncertainties."""
        return self._sigma_in

    def get_ages(self) -> np.ndarray:
        """Get array of grain ages."""
        return np.array([grain.age for grain in self.grains])

    def get_uncertainties(self, sigma: Optional[int] = None) -> np.ndarray:
        """
        Get array of grain uncertainties at specified sigma level.

        Parameters
        ----------
        sigma : int, optional
            Target sigma level. If None, returns raw uncertainties.
        """
        if sigma is None:
            return np.array([grain.uncertainty for grain in self.grains])
        return np.array([grain.get_uncertainty(sigma) for grain in self.grains])

    def get_uncertainties_1s(self) -> np.ndarray:
        """Get array of grain uncertainties normalized to 1-sigma."""
        return np.array([grain.uncertainty_1s for grain in self.grains])

    def get_uncertainties_2s(self) -> np.ndarray:
        """Get array of grain uncertainties at 2-sigma."""
        return np.array([grain.uncertainty_2s for grain in self.grains])

    def replace_grain_uncertainties(self, bandwidth: float):
        for grain in self.grains:
            grain._uncertainty_raw = bandwidth
        return self

    def get_q1_age(self):
        ages = self.get_ages()
        q1_age = np.quantile(ages, 0.25)
        return q1_age

    def get_median_age(self):
        ages = self.get_ages()
        median_age = np.quantile(ages, 0.5)
        return median_age

    def get_q3_age(self):
        ages = self.get_ages()
        q3_age = np.quantile(ages, 0.75)
        return q3_age

    def get_outlier_grains(self):
        q1 = self.get_q1_age()
        q3 = self.get_q3_age()
        iqr = q3 - q1
        outliers = []
        for grain in self.grains:
            if grain.age > q3 + 1.5 * iqr or grain.age < q1 - 1.5 * iqr:
                outliers.append(grain)
        return outliers

    def to_dict(self):
        return {
            'name': self.name,
            'grains': [grain.to_dict() for grain in self.grains],
            'sigma_in': self._sigma_in
        }

    def subset(self, min_age: float, max_age: float, uncertainty_coefficient: float = 0):
        subset_grains = []
        for grain in self.grains:
            if grain.age - (grain.uncertainty * uncertainty_coefficient) >= min_age:
                if grain.age + (grain.uncertainty * uncertainty_coefficient) <= max_age:
                    subset_grains.append(grain)
        return Sample(self.name, subset_grains, sigma_in=self._sigma_in)

    def to_mda_sample(self):
        """
        Convert to MDA module Sample for MDA calculations.

        Returns
        -------
        mda.Sample
            Sample object compatible with MDA functions.
        """
        from dz_lib.univariate.mda import Sample as MDASample
        ages = self.get_ages()
        errs = self.get_uncertainties_1s()  # MDA expects 1-sigma internally
        return MDASample(ages, errs, sigma_in=1, name=self.name)
