from email.charset import add_alias

import numpy as np


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

    associations += learning_rate * belief * prediction_error

    return associations

# Initialize association strengths as a NumPy array
associations = np.zeros(3, float)

# Define parameters
learning_rate = 0.1

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