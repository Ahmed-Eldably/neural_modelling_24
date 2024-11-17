import numpy as np
import matplotlib.pyplot as plt

class TemporalDifferenceLearning:
    """
    A class for Temporal Difference Learning to predict future rewards.
    """
    def __init__(self, n_time_steps, alpha, gamma=0.9):
        """
        Initialize the TD learning model.

        Parameters:
        - n_time_steps (int): Total number of time steps per trial.
        - alpha (float): Learning rate (alpha).
        - gamma (float): Discount factor for future rewards (gamma).
        """
        self.n_time_steps = n_time_steps
        self.alpha = alpha
        self.gamma = gamma
        self.V = np.zeros(n_time_steps)  # Prediction vector V(t)
        self.delta = np.zeros(n_time_steps)  # Temporal Difference Error (delta)

    def update(self, reward_vector, stimulus_vector=None):
        """
        Perform one trial of TD learning.

        Parameters:
        - reward_vector (np.array): Reward signal (r(t)) for each time step.
        - stimulus_vector (np.array): Stimulus signal (u(t)) for each time step (optional).
        """
        V_prev = np.copy(self.V)  # Store previous predictions

        # Temporal Difference Update over time steps
        for t in range(self.n_time_steps - 1):
            # Calculate the temporal difference error δ(t)
            self.delta[t] = reward_vector[t] + self.gamma * V_prev[t + 1] - V_prev[t]

            # Update predictions V(t) using TD learning rule
            self.V[t] += self.alpha * self.delta[t]

        # Immediate adjustment for stimulus influence (if provided)
        if stimulus_vector is not None:
            for t in range(self.n_time_steps):
                if stimulus_vector[t] > 0:
                    # Directly update V(t) where stimulus is present to ensure learning influence
                    self.V[t] += self.alpha * (1.0 - self.V[t])

    def reset(self):
        """Reset predictions and errors for a new set of trials."""
        self.V = np.zeros(self.n_time_steps)
        self.delta = np.zeros(self.n_time_steps)

    def get_predictions(self):
        """Return the current prediction vector."""
        return self.V

    def get_temporal_difference_error(self):
        """Return the temporal difference error vector."""
        return self.delta


# Re-test with the corrected implementation
n_time_steps = 300
alpha = 0.1
gamma = 0.9
td_model = TemporalDifferenceLearning(n_time_steps, alpha, gamma)

# Stimulus and reward vectors
stimulus_vector = np.zeros(n_time_steps)
reward_vector = np.zeros(n_time_steps)
stimulus_time = 100
reward_time = 200
stimulus_vector[stimulus_time] = 1
reward_vector[reward_time] = 1

# Reset the model
td_model.reset()

for trial in range(100):
    td_model.update(reward_vector, stimulus_vector)

# Fetch predictions and errors after all trials
predictions = td_model.get_predictions()
errors = td_model.get_temporal_difference_error()

# Display updated predictions and errors near stimulus and reward
print("Updated Predictions (V) after 10 trials (around stimulus):", predictions[90:110])
print("Updated Predictions (V) after 10 trials (around reward):", predictions[190:210])
print("\nUpdated Errors (δ) after 10 trials (around stimulus):", errors[90:110])
print("Updated Errors (δ) after 10 trials (around reward):", errors[190:210])

# Easy values to test
# Manually setting reward and stimulus to check correct propagation
stimulus_vector = np.zeros(n_time_steps)
reward_vector = np.zeros(n_time_steps)
stimulus_vector[50] = 1  # Set a simple stimulus at t=50
reward_vector[100] = 1  # Set a reward at t=100

# Reset the model for a simple test
td_model.reset()

# Run a single trial with the simple stimulus and reward
td_model.update(reward_vector, stimulus_vector)

# Fetch predictions and errors for inspection
predictions = td_model.get_predictions()
errors = td_model.get_temporal_difference_error()

# Display predictions and errors near test values
print("\nSimple Test - Predictions (V) around stimulus at t=50:", predictions[40:60])
print("Expected Prediction (V) at t=50: 0.1")

