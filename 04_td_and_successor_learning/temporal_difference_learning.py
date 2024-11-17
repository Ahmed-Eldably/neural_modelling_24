import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TemporalDifferenceLearning:
    """
    A class for Temporal Difference Learning to predict future rewards.
    """
    def __init__(self, n_time_steps, alpha, gamma=0.9):
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha (learning rate) must be between 0 and 1.")
        if not (0 <= gamma <= 1):
            raise ValueError("Gamma (discount factor) must be between 0 and 1.")
        if n_time_steps <= 0:
            raise ValueError("n_time_steps must be a positive integer.")

        self.n_time_steps = n_time_steps
        self.alpha = alpha
        self.gamma = gamma
        self.V = np.zeros(n_time_steps)  # Prediction vector V(t)
        self.delta = np.zeros(n_time_steps)  # Temporal Difference Error (delta)
        self.predictions_over_trials = []  # Store predictions over multiple trials
        self.delta_over_trials = []  # Store delta values over multiple trials

    def update(self, reward_vector, stimulus_vector=None):
        V_prev = np.copy(self.V)  # Store previous predictions

        for t in range(self.n_time_steps):
            # Temporal Difference Update
            if t < self.n_time_steps - 1:
                self.delta[t] = reward_vector[t] + self.gamma * V_prev[t + 1] - V_prev[t]
                self.V[t] += self.alpha * self.delta[t]

            # Immediate adjustment for stimulus influence if applicable
            if stimulus_vector is not None and stimulus_vector[t] > 0:
                self.V[t] += self.alpha * (1.0 - self.V[t])

    def reset(self):
        self.V = np.zeros(self.n_time_steps)
        self.delta = np.zeros(self.n_time_steps)

    def run_trials(self, reward_vector, stimulus_vector, num_trials):
        if len(reward_vector) != self.n_time_steps or (stimulus_vector is not None and len(stimulus_vector) != self.n_time_steps):
            raise ValueError("The length of reward_vector and stimulus_vector must match n_time_steps.")

        self.predictions_over_trials = []
        self.delta_over_trials = []

        for trial in range(num_trials):
            self.update(reward_vector, stimulus_vector)
            self.predictions_over_trials.append(self.get_predictions().copy())
            self.delta_over_trials.append(self.get_temporal_difference_error().copy())

    def get_predictions(self):
        return self.V

    def get_temporal_difference_error(self):
        return self.delta

    def plot_3d_predictions(self, cmap='viridis'):
        if not self.predictions_over_trials:
            raise ValueError("No trials have been run. Please run trials before plotting.")

        predictions_over_trials = np.array(self.predictions_over_trials)
        num_trials = len(self.predictions_over_trials)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        t = np.arange(self.n_time_steps)
        trials = np.arange(num_trials)
        t, trials = np.meshgrid(t, trials)

        ax.plot_surface(t, trials, predictions_over_trials, cmap=cmap)
        ax.set_title("3D Visualization of Predictions (V) over Trials")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Trials")
        ax.set_zlabel("Value (V)")

        plt.tight_layout()
        plt.show()

    def plot_3d_delta(self, cmap='plasma'):
        if not self.delta_over_trials:
            raise ValueError("No trials have been run. Please run trials before plotting.")

        delta_over_trials = np.array(self.delta_over_trials)
        num_trials = len(self.delta_over_trials)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        t = np.arange(self.n_time_steps)
        trials = np.arange(num_trials)
        t, trials = np.meshgrid(t, trials)

        ax.plot_surface(t, trials, delta_over_trials, cmap=cmap)
        ax.set_title("3D Visualization of Temporal Difference Error (Delta) over Trials")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Trials")
        ax.set_zlabel("TD Error (Delta)")

        plt.tight_layout()
        plt.show()

