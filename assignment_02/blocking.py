import matplotlib.pyplot as plt


class BlockingParadigm:
    def __init__(self, alpha=0.1, lambda_=1.0, initial_strengths=None):
        """
        Parameters:
        alpha (float): Learning rate
        lambda_ (float): Maximum associative strength (reward value)
        initial_strengths (dict): Initial associative strengths for each stimulus
        """
        self.alpha = alpha
        self.lambda_ = lambda_
        self.V = initial_strengths or {"S1": 0.0, "S2": 0.0}
        self.history = {stimulus: [] for stimulus in self.V}

    def _update(self, present_stimuli, reward):
        """
        Apply the Rescorla-Wagner update for a single stimuli

        Parameters:
        present_stimuli (list of str): Stimuli present in the current trial
        reward (float): Reward value for the current trial
        """
        V_total = sum(self.V[stimulus] for stimulus in present_stimuli)
        delta_V = self.alpha * (reward - V_total)

        for stimulus in present_stimuli:
            self.V[stimulus] += delta_V
            self.history[stimulus].append(self.V[stimulus])

    def start_pre_training(self, trials):
        """Pre-Training Phase: Only S1 is paired with reward."""
        for _ in range(trials):
            self._update(present_stimuli=["S1"], reward=1)
            self.history["S2"].append(self.V["S2"])

    def start_training(self, trials):
        """Training Phase: Both S1 and S2 are presented with reward."""
        for _ in range(trials):
            self._update(present_stimuli=["S1", "S2"], reward=1)

    def plot_history(self):
        """Plot the history of associative strengths for each stimulus."""
        for stimulus, values in self.history.items():
            plt.plot(values, label=f"Associative Strength of {stimulus}")
        plt.xlabel("Trial")
        plt.ylabel("Associative Strength (V)")
        plt.legend()
        plt.title("Blocking Effect in Rescorla-Wagner Model")
        plt.show()


# Initialize and run the blocking paradigm simulation
blocking_paradigm = BlockingParadigm(alpha=0.1, lambda_=1.0)

# Define number of trials for each phase
trials_pre_training = 50
trials_training = 50

blocking_paradigm.start_pre_training(trials=trials_pre_training)
blocking_paradigm.start_training(trials=trials_training)
blocking_paradigm.plot_history()
