import numpy as np
import matplotlib.pyplot as plt


# Configuration Parameters
conditioning_trials = 50
extinction_trials = 50
test_trials = 1
total_trials = conditioning_trials + extinction_trials + test_trials
learning_rate = 0.1
similarity_threshold = 0.5
delay_threshold = 10

# Part 1: Designing an appropriate array for the animal to update beliefs (condition - extinction - test)
def initialize_beliefs(conditioning_trials, extinction_trials, test_trials):

    # Initialize the belief array
    belief_array = np.zeros((conditioning_trials + extinction_trials + test_trials, 3))

    belief_array[:conditioning_trials, 0] = 1 # State 1 belief during conditioning phase
    belief_array[conditioning_trials:conditioning_trials+extinction_trials, 1] = 1 # State 2 belief during extinction phase
    belief_array[-1, 2] = 1 # State 3 belief during test trial

    return belief_array

# Part 2.1: A numpy array for expectations based on the states (condition - extinction - test)
def initialize_expectations(conditioning_trials, extinction_trials, test_trials):

    # Mapping the expectation to each trial
    expectation = np.zeros(conditioning_trials + extinction_trials + test_trials)
    expectation[:conditioning_trials] = 1.0 # High expectation during Conditioning
    expectation[conditioning_trials:conditioning_trials + extinction_trials] = 0.0 # Low expectation during Extinction
    expectation[-1] = 0.5 # Moderate expectation during Delayed Test

    return expectation
# Part 2.2: Plotting expectation
def plot_expectation(expectation):
    """
    Plot the expectation of receiving the US across trials.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(expectation, label="Expectation of US after CS")
    plt.xlabel("Trial")
    plt.ylabel("Expectation of US")
    plt.title("Animal's Expectation of US Across Trials")
    plt.legend()
    plt.grid(True)
    plt.show()

# Part 3: Inferring beliefs based on the previous state, similarity, and time gap
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
            belief[1] = 1 - similarity # Shift to extinction state if from conditioning
        elif prev_state == 2:
            belief[2] = 1 - similarity # Shift to test state if from extinction

    # Adjustment based on delay
    if delay > delay_threshold:
        # If the delay is long, introduces uncertainty, it favors the test state
        belief[2] += delay / (delay + delay_threshold)

    # Normalize belief distribution so it sums to 1
    belief = belief / belief.sum() if belief.sum() > 0 else belief

    return belief


# Part 4: Using the RW model the update association strength
def update_association_strengths(associations, belief, learning_rate, reward):
    """
    Parameters:
    - associations (np.array): current association strengths for each state.
    - belief (np.array): Current belief distribution over states.
    - learning_rate (float): learning rate for the association update (between 0.0 and 1.0).
    - reward (float): the actual reward (US) for the current trial.

    :return:
    - association_strengths: updated association strengths for each state.
    """

    prediction = np.dot(associations, belief)
    prediction_error = reward - prediction

    delta = learning_rate * belief * prediction_error
    associations += delta
    return associations


# Initialize belief array and expectation array
belief_array = initialize_beliefs(conditioning_trials, extinction_trials, test_trials)
expectation = initialize_expectations(conditioning_trials, extinction_trials, test_trials)

# Print belief array and expectation shapes for verification
print("Belief array shape:", belief_array.shape)
print("Expectation shape:", expectation.shape)

# Plot expectation
plot_expectation(expectation)

# Initialize association strengths
associations = np.zeros(3, float)


# Trial sequence demonstrating conditioning, extinction, and test phases
# Trial 1: Conditioning trial (high similarity, low delay)
belief = infer_belief(1, similarity=0.8, delay=2)  # High similarity, low delay -> Strong belief in State 1
reward = 1  # US presented
print("Before Trial 1:", associations)
associations = update_association_strengths(associations, belief, learning_rate, reward)
print("After Trial 1:", associations)

# Trial 2: Extinction trial (low similarity, low delay)
belief = infer_belief(1, similarity=0.3, delay=2)  # Low similarity, low delay -> Shift to State 2
reward = 0  # No US
print("Before Trial 2:", associations)
associations = update_association_strengths(associations, belief, learning_rate, reward)
print("After Trial 2:", associations)

# Trial 3: Test trial (low similarity, high delay)
belief = infer_belief(2, similarity=0.3, delay=15)  # Low similarity, high delay -> Shift to State 3
reward = 1  # Test trial with US
print("Before Trial 3:", associations)
associations = update_association_strengths(associations, belief, learning_rate, reward)
print("After Trial 3:", associations)

# Final summary of associations after all trials
print("Final associative strengths:", associations)

