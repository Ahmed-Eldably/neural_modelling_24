import numpy as np
import matplotlib.pyplot as plt

belief_array = np.zeros(101, dtype=int)

belief_array[:50] = 1  # State 1: Conditioning
belief_array[50:100] = 2  # State 2: Extinction
belief_array[100] = 3  # State 3: Delayed Test

print(belief_array)

# Mapping the expectations to each trial
expectations = np.zeros(101, dtype=float)
expectations[belief_array == 1] = 1.0 # High expectations during Conditioning
expectations[belief_array == 2] = 0.0 # Low expectations during Extinction
expectations[belief_array == 3] = 0.5 # Moderate expectations during Delayed Test


# Plotting the expectation
plt.figure(figsize=(10, 5))
plt.plot(expectations, label="Expectation of US after CS")
plt.xlabel("Trial")
plt.ylabel("Expectation of US")
plt.title("Expectation of Receiving US for Each Trial")
plt.legend()
plt.grid(True)
plt.show()

def infer_belief(prev_state, similarity, time_gap):
    """
    :return:
    """
    pass
