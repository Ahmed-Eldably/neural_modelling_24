import numpy as np
import matplotlib.pyplot as plt


conditioning_trials = 50
extinction_trials = 50
test_trials = 1

# Initialize the belief array
belief_array = np.zeros((conditioning_trials + extinction_trials + test_trials, 3))

belief_array[:conditioning_trials, 0] = 1 # State 1 belief during conditioning phase
belief_array[conditioning_trials:conditioning_trials+extinction_trials, 1] = 1 # State 2 belief during extinction phase
belief_array[-1, 2] = 1 # State 3 belief during test trial

print(belief_array.shape)
print(belief_array)

# Mapping the expectation to each trial
expectation = np.zeros(conditioning_trials + extinction_trials + test_trials)
expectation[:conditioning_trials] = 1.0 # High expectation during Conditioning
expectation[conditioning_trials:conditioning_trials + extinction_trials] = 0.0 # Low expectation during Extinction
expectation[-1] = 0.5 # Moderate expectation during Delayed Test

print(expectation.shape)
print(expectation)


# Plotting the expectation
plt.figure(figsize=(10, 5))
plt.plot(expectation, label="Expectation of US after CS")
plt.xlabel("Trial")
plt.ylabel("Expectation of US")
plt.title("Animal's Expectation of US Across Trials")
plt.legend()
plt.grid(True)
plt.show()

def infer_belief(prev_state, similarity, delay):
    """
    Parameters:
    - prev_state (int): the last inferred state
    - similarity (float): representing how similar the previous state and the current state (the value is between 0 and 1)
    - delay (int): indicating the time passed since the last trial. (In days)

    :return:
    - belief (numpy array): the belief distribution array over states [P(state 1), P(state 2), P(State 3)].
    """
    belief = np.zeros(3, float)

    similarity_threshold = 0.5
    delay_threshold = 10

    if similarity > similarity_threshold:
        # If similarity is high, it favors the previous state
        belief[prev_state - 1] = 1 - delay / (delay + delay_threshold)  # Decay function
    else:
        # If similarity is low, it favors a shift to a different state (extinction or something new)
        if prev_state == 1:
            belief[1] = 1 - similarity_threshold # Shift to extinction state if from conditioning
        elif prev_state == 2:
            belief[2] = 1 - similarity_threshold # Shift to test state if from extinction

    # Adjustment based on delay
    if delay > delay_threshold:
        # If the delay is long, introduces uncertainty, it favors the test state
        belief[2] += delay / (delay + delay_threshold)
    else:
        # Normalize belief distribution so it sums to 1
        belief = belief.sum() if belief.sum() > 0 else belief

    return belief



