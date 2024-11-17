import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        self.predictions_over_trials = []  # Store predictions over multiple trials
        self.delta_over_trials = []  # Store delta values over multiple trials

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

    def run_trials(self, reward_vector, stimulus_vector, num_trials):
        """
        Run multiple trials to observe learning progression.

        Parameters:
        - reward_vector (np.array): Reward signal (r(t)) for each time step.
        - stimulus_vector (np.array): Stimulus signal (u(t)) for each time step.
        - num_trials (int): Number of trials to run.
        """
        self.predictions_over_trials = []
        self.delta_over_trials = []
        for trial in range(num_trials):
            self.update(reward_vector, stimulus_vector)
            self.predictions_over_trials.append(np.copy(self.get_predictions()))
            self.delta_over_trials.append(np.copy(self.get_temporal_difference_error()))

    def plot_3d_predictions(self):
        """
        Plot a 3D visualization of the predictions over multiple trials.
        """
        if not self.predictions_over_trials:
            print("No trials have been run. Please run trials before plotting.")
            return

        predictions_over_trials = np.array(self.predictions_over_trials)
        num_trials = len(self.predictions_over_trials)

        # 3D Visualization of Predictions (V) over Trials
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create meshgrid for trials and time steps
        T = np.arange(self.n_time_steps)
        Trials = np.arange(num_trials)
        T, Trials = np.meshgrid(T, Trials)

        # Plot Predictions over Time Steps
        ax.plot_surface(T, Trials, predictions_over_trials, cmap='viridis')
        ax.set_title("3D Visualization of Predictions (V) over Trials")
        ax.set_xlabel("Time Steps")  # Switch X-axis label to Time Steps
        ax.set_ylabel("Trials")       # Switch Y-axis label to Trials
        ax.set_zlabel("Value (V)")
        ax.zaxis.labelpad = 15  # Adjust the labelpad to move the label to the left

        plt.tight_layout()
        plt.show()

    def plot_3d_delta(self):
        """
        Plot a 3D visualization of the temporal difference error (delta) over multiple trials.
        """
        if not self.delta_over_trials:
            print("No trials have been run. Please run trials before plotting.")
            return

        delta_over_trials = np.array(self.delta_over_trials)
        num_trials = len(self.delta_over_trials)

        # 3D Visualization of Delta (TD Error) over Trials
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Create meshgrid for trials and time steps
        T = np.arange(self.n_time_steps)
        Trials = np.arange(num_trials)
        T, Trials = np.meshgrid(T, Trials)

        # Plot Delta over Time Steps
        ax.plot_surface(T, Trials, delta_over_trials, cmap='plasma')
        ax.set_title("3D Visualization of Temporal Difference Error (Delta) over Trials")
        ax.set_xlabel("Time Steps")  # Switch X-axis label to Time Steps
        ax.set_ylabel("Trials")       # Switch Y-axis label to Trials
        ax.set_zlabel("TD Error (Delta)")
        ax.zaxis.labelpad = 15  # Adjust the labelpad to move the label to the left

        plt.tight_layout()
        plt.show()

    def plot_detailed_figure(self, stimulus_vector, reward_vector):
        """
        Plot the detailed figure as described in Figure 9.2B with clearer visualization.
        """
        fig, axes = plt.subplots(5, 2, figsize=(25, 25), sharex=True)

        time = np.arange(self.n_time_steps)
        fontsize = 25

        # Plot all values before training
        axes[0, 0].plot(time, stimulus_vector, label='Stimulus u(t)', color='b')
        axes[0, 0].set_ylabel('Stimulus u(t)', fontsize=fontsize)
        axes[0, 0].set_title('Before Training', fontsize=fontsize)
        axes[0, 0].legend(fontsize=fontsize)

        axes[1, 0].plot(time, reward_vector, label='Reward r(t)', color='g')
        axes[1, 0].set_ylabel('Reward r(t)', fontsize=fontsize)
        axes[1, 0].legend(fontsize=fontsize)

        axes[2, 0].plot(time, np.zeros(self.n_time_steps), label='Prediction v(t) before training', linestyle='--')
        axes[2, 0].set_ylabel('Prediction v(t)', fontsize=fontsize)
        axes[2, 0].legend(fontsize=fontsize)

        Delta_v_before = np.diff(np.zeros(self.n_time_steps), prepend=0)
        axes[3, 0].plot(time[:-1], Delta_v_before[:-1], label='Delta v(t-1) before training', linestyle='--')
        axes[3, 0].set_ylabel('Delta v(t-1)', fontsize=fontsize)
        axes[3, 0].legend(fontsize=fontsize)

        delta_before = reward_vector[:-1] + Delta_v_before[:-1]
        axes[4, 0].plot(time[:-1], delta_before, label='Delta (t-1) before training', linestyle='--')
        axes[4, 0].set_ylabel('Delta(t-1)', fontsize=fontsize)
        axes[4, 0].set_xlabel('Time t', fontsize=fontsize)
        axes[4, 0].legend(fontsize=fontsize)

        # Plot all values after training
        axes[0, 1].plot(time, stimulus_vector, label='Stimulus u(t)', color='b')
        axes[0, 1].set_ylabel('Stimulus u(t)', fontsize=fontsize)
        axes[0, 1].set_title('After Training', fontsize=fontsize)
        axes[0, 1].legend(fontsize=fontsize)

        axes[1, 1].plot(time, reward_vector, label='Reward r(t)', color='g')
        axes[1, 1].set_ylabel('Reward r(t)', fontsize=fontsize)
        axes[1, 1].legend(fontsize=fontsize)

        axes[2, 1].plot(time, self.get_predictions(), label='Prediction v(t) after training', linestyle='-', color='r')
        axes[2, 1].set_ylabel('Prediction v(t)', fontsize=fontsize)
        axes[2, 1].legend(fontsize=fontsize)

        Delta_v_after = np.diff(self.get_predictions(), prepend=0)
        axes[3, 1].plot(time[:-1], Delta_v_after[:-1], label='Delta v(t-1) after training', linestyle='-', color='r')
        axes[3, 1].set_ylabel('Delta v(t-1)', fontsize=fontsize)
        axes[3, 1].legend(fontsize=fontsize)

        delta_after = reward_vector[:-1] + Delta_v_after[:-1]
        axes[4, 1].plot(time[:-1], delta_after, label='Delta (t-1) after training', linestyle='-', color='r')
        axes[4, 1].set_ylabel('Delta(t-1)', fontsize=fontsize)
        axes[4, 1].set_xlabel('Time t', fontsize=fontsize)
        axes[4, 1].legend(fontsize=fontsize)

        plt.tight_layout()
        plt.show()




