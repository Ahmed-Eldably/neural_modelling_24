import numpy as np


def simulate_blocking(arr_size, number_of_stimulus):
    stimulus = np.full((arr_size, number_of_stimulus), fill_value=[1, 0])
    print(stimulus.shape)
    training = np.full(shape=(arr_size, number_of_stimulus), fill_value=[1, 1])
    print(training.shape)
    rewards = np.array(np.ones(arr_size))
    print(rewards.shape)

    print(stimulus)
    print(f"\n{training}")
    print(f"\n{rewards}")


arr_size = 75
number_of_stimulus = 2
print(simulate_blocking(
    arr_size=arr_size,
    number_of_stimulus=number_of_stimulus
))

