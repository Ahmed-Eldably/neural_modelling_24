from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class ClassicalConditioningParadigm(ABC):
    def __init__(self, alpha=0.1, lambda_=1.0, initial_strengths=None):
        """
        Initialize common model parameters.

        Parameters:
        alpha (float): Learning rate
        lambda_ (float): Maximum associative strength (reward value)
        initial_strengths (dict): Initial associative strengths for stimuli
        """
        self.alpha = alpha
        self.lambda_ = lambda_
        self.V = initial_strengths or {"S1": 0.0, "S2": 0.0}
        self.history = {stimulus: [] for stimulus in self.V}

    def _update(self, present_stimuli, reward):
        """Generic Rescorla-Wagner update function."""
        V_total = sum(self.V[stimulus] for stimulus in present_stimuli)
        delta_V = self.alpha * (reward - V_total)

        for stimulus in present_stimuli:
            self.V[stimulus] += delta_V
            self.history[stimulus].append(self.V[stimulus])

    @abstractmethod
    def start_pre_training(self, trials):
        """Define the pre-training phase for each paradigm."""
        pass

    @abstractmethod
    def start_training(self, trials):
        """Define the training phase for each paradigm."""
        pass

    def plot_history(self, title="Conditioning Paradigm"):
        """Plot the history of associative strengths."""
        for stimulus, values in self.history.items():
            plt.plot(values, label=f"Associative Strength of {stimulus}")
        plt.xlabel("Trial")
        plt.ylabel("Associative Strength (V)")
        plt.legend()
        plt.title(title)
        plt.show()