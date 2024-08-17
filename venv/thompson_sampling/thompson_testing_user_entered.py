import Thompson_sampling as ts
import numpy as np
from Bandit import Bandit

# print("Hello! Let's test a UCB algorithm implementaion on the multi-armed bandits problem!")
num_bandits = int(input("How many bandits would you like to test?   "))

bandits = []

want = input('would you like to input vals?  (yes/no)  ')

if want.__eq__('yes'):

    for i in range(num_bandits):
        probability = float( input(f'Probability of Bandit{i}: ') )
        bandits.append(Bandit(probability))


    num_samples = int( input("Total number of samples to iterate through: ") )
    samples = []

    for i in range(num_samples):
        sample_size = int( input(f'Sample #{i+1}: ') )
        samples.append(sample_size)

else:
    for i in range(num_bandits):
        probability = np.random.rand()
        bandits.append(Bandit(probability))


    samples = [1000, 2000]



ts.play(bandits, samples)

# ts.play_average(bandits, samples, .7)