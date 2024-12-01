import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TemporalDifferenceLearning:
    """
    Temporal Difference Learning implementation for predicting delayed rewards.
    """

    def __init__(self, n_time_steps, alpha, gamma=0.9):
        """
        Initialize the TD learning model.

        Parameters:
        - n_time_steps (int): Total number of time steps in a trial.
        - alpha (float): Learning rate.
        - gamma (float): Discount factor for future rewards.
        """
        self.n_time_steps = n_time_steps
        self.alpha = alpha
        self.gamma = gamma
        self.V = np.zeros(n_time_steps)  # Initialize value predictions
        self.delta = np.zeros(n_time_steps)  # Temporal difference errors

    def update(self, reward_vector):
        """
        Perform a single trial of TD learning.

        Parameters:
        - reward_vector (np.array): Reward signal (r(t)) for each time step.
        """
        V_prev = np.copy(self.V)  # Store previous predictions
        for t in range(self.n_time_steps - 1):
            # Temporal Difference Error δ(t)
            self.delta[t] = reward_vector[t] + self.gamma * V_prev[t + 1] - V_prev[t]
            # Update value predictions V(t)
            self.V[t] += self.alpha * self.delta[t]

    def run_trials(self, reward_time, stimulus_time, num_trials):
        """
        Simulate multiple trials of TD learning.

        Parameters:
        - reward_time (int): Time step when the reward is delivered.
        - stimulus_time (int): Time step when the stimulus is presented.
        - num_trials (int): Number of trials to simulate.

        Returns:
        - predictions_over_trials: List of value predictions V(t) over trials.
        - delta_over_trials: List of temporal difference errors δ(t) over trials.
        """
        predictions_over_trials = []
        delta_over_trials = []

        reward_vector = np.zeros(self.n_time_steps)
        reward_vector[reward_time] = 1.0  # Reward delivered at reward_time

        for trial in range(num_trials):
            self.update(reward_vector)
            predictions_over_trials.append(np.copy(self.V))
            delta_over_trials.append(np.copy(self.delta))

        return predictions_over_trials, delta_over_trials

    def plot_results(self, predictions_over_trials, delta_over_trials, stimulus_time, reward_time):
        """
        Plot results to reproduce Figure 9.2 from the task.

        Parameters:
        - predictions_over_trials (list): Predictions V(t) over trials.
        - delta_over_trials (list): Temporal difference errors δ(t) over trials.
        - stimulus_time (int): Time step when the stimulus is presented.
        - reward_time (int): Time step when the reward is delivered.
        """
        num_trials = len(predictions_over_trials)
        T = np.arange(self.n_time_steps)
        Trials = np.arange(num_trials)

        # 3D Surface Plot of Temporal Difference Errors (δ)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        T_mesh, Trials_mesh = np.meshgrid(T, Trials)
        delta_surface = np.array(delta_over_trials)
        ax.plot_surface(T_mesh, Trials_mesh, delta_surface, cmap='viridis')
        ax.set_title("Temporal Difference Error (δ) Over Trials")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Trials")
        ax.set_zlabel("δ(t)")
        plt.show()

        # Detailed Plots
        time = np.arange(self.n_time_steps)
        predictions = predictions_over_trials[-1]

        fig, axes = plt.subplots(5, 2, figsize=(15, 20), sharex=True)
        fontsize = 12

        # Left Column: Before Training
        axes[0, 0].plot(time, np.zeros(self.n_time_steps), label="Stimulus u(t)", linestyle='--')
        axes[1, 0].plot(time, np.zeros(self.n_time_steps), label="Reward r(t)", linestyle='--')
        axes[2, 0].plot(time, np.zeros(self.n_time_steps), label="Prediction V(t)", linestyle='--')
        axes[3, 0].plot(time[:-1], np.zeros(self.n_time_steps - 1), label="∆V(t-1)", linestyle='--')
        axes[4, 0].plot(time[:-1], np.zeros(self.n_time_steps - 1), label="TD Error δ(t-1)", linestyle='--')
        axes[0, 0].set_title("Before Training", fontsize=fontsize)

        # Right Column: After Training
        axes[0, 1].axvline(stimulus_time, color='b', label="Stimulus u(t)", linestyle='--')
        axes[1, 1].axvline(reward_time, color='g', label="Reward r(t)", linestyle='--')
        axes[2, 1].plot(time, predictions, label="Prediction V(t)", color='r')
        delta_v = np.diff(predictions, prepend=0)
        axes[3, 1].plot(time, delta_v, label="∆V(t-1)", color='r')
        axes[4, 1].plot(time, delta_v, label="TD Error δ(t-1)", color='r')
        axes[0, 1].set_title("After Training", fontsize=fontsize)

        for ax in axes.ravel():
            ax.legend(fontsize=fontsize)
            ax.tick_params(axis='both', labelsize=fontsize)

        axes[4, 0].set_xlabel("Time Steps", fontsize=fontsize)
        axes[4, 1].set_xlabel("Time Steps", fontsize=fontsize)

        plt.tight_layout()
        plt.show()


# Example
if __name__ == "__main__":
    n_time_steps = 200
    alpha = 0.1
    gamma = 0.9
    num_trials = 50
    stimulus_time = 100
    reward_time = 150

    td = TemporalDifferenceLearning(n_time_steps, alpha, gamma)
    predictions, deltas = td.run_trials(reward_time, stimulus_time, num_trials)
    td.plot_results(predictions, deltas, stimulus_time, reward_time)
