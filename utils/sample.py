class Sample:
    def __init__(self, name, grains):
        self.name = name
        self.grains = grains

    def replace_bandwidth(self, new_bandwidth):
        for grain in self.grains:
            grain.uncertainty = new_bandwidth
        return self

    def get_oldest_grain(self):
        grains = sorted(self.grains, key=lambda grain: grain.age, reverse=True)
        return grains[0]

    def get_youngest_grain(self):
        grains = sorted(self.grains, key=lambda grain: grain.age, reverse=False)
        return grains[0]

class Grain:
    def __init__(self, age, uncertainty):
        self.age = age
        self.uncertainty = uncertainty
