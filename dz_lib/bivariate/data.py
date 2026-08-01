import numpy as np

class BivariateGrain:
    def __init__(self, age: float, hafnium: float):
        self.age = age
        self.hafnium = hafnium

    def to_dict(self):
        return {
            'age': self.age,
            'hafnium': self.hafnium
        }

class BivariateSample:
    def __init__(self, name: str, grains: [BivariateGrain]):
        self.name = name
        self.grains = grains

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

    def get_ages(self):
        grains = self.grains
        ages = [grain.age for grain in grains]
        return ages

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
            'grains': [grain.to_dict() for grain in self.grains]
        }