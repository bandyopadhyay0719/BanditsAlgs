class Bandit:

    def __init__(self, probability):
        self.probability = probability

        self.success_fail = [0,0]

        self.vals = []



    def __str__(self):
        return f"probability: {self.probability}, successes: {self.success_fail[0]}, fails: {self.success_fail[1]} \n vals: {self.vals}"