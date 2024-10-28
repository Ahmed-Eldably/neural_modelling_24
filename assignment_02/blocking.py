import numpy as np
import matplotlib.pyplot as plt


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


def plot_blocker():
    expectations = 1 - np.exp(-0.1 * np.arange(2 * arr_size))
    plt.plot(expectations)
    plt.xlabel("Trial")
    plt.ylabel("Expected reward")
    plt.show()


x = np.arange(0, 75)
print(x)
print(x.shape)

# Define the functions
expectations = 1 - np.exp(-0.1 * np.arange(75))
rewards = np.array(np.ones(75))

# Plot both functions
plt.plot(x, expectations, label="predictions", color="blue")
plt.plot(x, rewards, label="rewards", color="red")

# Add titles and labels
plt.xlabel("trials")
plt.ylabel("Rewards")

# Add a legend
plt.legend()

# Show the plot
plt.show()

arr_size = 75
number_of_stimulus = 2
print(simulate_blocking(
    arr_size=arr_size,
    number_of_stimulus=number_of_stimulus
))


