import numpy as np

class Bandit:

    def __init__(self, probability):
        self.probability = probability

        self.vals = []

    def __str__(self):
        return f"probability: {self.probability}, vals: {self.vals}"

