class Sample:
    def __init__(self, name, grains):
        self.name = name
        self.grains = grains

    def replace_bandwidth(self, new_bandwidth):
        for grain in self.grains:
            grain.uncertainty = new_bandwidth
        return self


class Grain:
    def __init__(self, age, uncertainty):
        self.age = age
        self.uncertainty = uncertainty
