
class Bandit:

    bandits = []
    def __init__(self, probability):
        self.probability = probability

        self.vals = []
        Bandit.bandits.append(self)

    def __str__(self):
        return f"probability: {self.probability}, vals: {self.vals}"