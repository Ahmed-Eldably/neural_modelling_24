from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class ClassicalConditioningParadigm(ABC):
    def __init__(self,
                 alpha1=0.1,
                 alpha2=0.05,
                 lambda_=1.0):
        """
        Parameters:
        - alpha1 (float): Learning rate for stimulus S1
        - alpha2 (float): Learning rate for stimulus S2
        - lambda_ (float): Maximum associative strength (reward value)
        """
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.lambda_ = lambda_
        self.V = {"S1": 0.0, "S2": 0.0}
        self.history = {"S1": [], "S2": []}

    def update_associative_strength(self, present_stimuli, reward):
        """
        Applies the Rescorla-Wagner rule to update associative strengths for each stimulus.

        Parameters:
        - present_stimuli (list): List of stimuli presented in the trial (e.g., ['S1', 'S2'])
        - reward (float): The reward value for the trial
        """
        # Calculate total associative strength prediction for all present stimuli
        v_total = sum(self.V[stimulus] for stimulus in present_stimuli)
        delta_v1 = self.alpha1 * (reward - v_total) if "S1" in present_stimuli else 0
        delta_v2 = self.alpha2 * (reward - v_total) if "S2" in present_stimuli else 0

        # Update associative strengths
        if "S1" in present_stimuli:
            self.V["S1"] += delta_v1
        if "S2" in present_stimuli:
            self.V["S2"] += delta_v2

        # Record the associative strength history for both stimuli on every trial
        self.history["S1"].append(self.V["S1"])
        self.history["S2"].append(self.V["S2"])

    @abstractmethod
    def pre_training(self, pre_training_trials=0):
        """Define the pre-training phase, if applicable."""
        pass

    @abstractmethod
    def training(self, training_trials=0):
        """Define the training phase specific to the paradigm."""
        pass

    def run(self,
            pre_training_trials=0,
            training_trials=0):
        """Run the pre-training (if any) and training phases sequentially."""
        self.pre_training(pre_training_trials=pre_training_trials)
        self.training(training_trials=training_trials)

    def plot_history(self, title="Conditioning Paradigm"):
        """Plot the associative strengths over trials."""
        for stimulus, values in self.history.items():
            plt.plot(values, label=f"Associative Strength of {stimulus}")
        plt.xlabel("Trial")
        plt.ylabel("Associative Strength (V)")
        plt.legend()
        plt.title(title)
        plt.show()