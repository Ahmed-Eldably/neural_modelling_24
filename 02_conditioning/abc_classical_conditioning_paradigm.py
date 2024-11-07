from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class ClassicalConditioningParadigm(ABC):
    def __init__(self,
                 learning_rate_1=0.1,
                 learning_rate_2=0.05,
                 max_reward=1.0):
        """
        Parameters:
        - learning_rate_1 (float): Learning rate for stimulus S1
        - learning_rate_2 (float): Learning rate for stimulus S2
        - max_reward (float): Maximum associative strength (target reward value)
        """
        self.learning_rate_1 = learning_rate_1
        self.learning_rate_2 = learning_rate_2
        self.max_reward = max_reward
        self.associative_strengths = {"S1": 0.0,
                                      "S2": 0.0}
        self.history = {"S1": [],
                        "S2": []}

    def update_associative_strength(self, present_stimuli, reward):
        """
        Applies the Rescorla-Wagner rule to update associative strengths for each stimulus.

        Parameters:
        - present_stimuli (list): List of stimuli presented in the trial (for now we only have S1 and S2)
        - reward (float): The reward value for the trial
        """
        # Calculate total prediction V(s)
        prediction = sum(self.associative_strengths[stimulus] for stimulus in present_stimuli)

        # Calculate prediction error
        prediction_error = reward - prediction

        # Calculate the update for each stimulus
        delta_w1 = self.learning_rate_1 * prediction_error if "S1" in present_stimuli else 0
        delta_w2 = self.learning_rate_2 * prediction_error if "S2" in present_stimuli else 0

        # Update associative strengths
        if "S1" in present_stimuli:
            self.associative_strengths["S1"] += delta_w1
        if "S2" in present_stimuli:
            self.associative_strengths["S2"] += delta_w2

        # Record the associative strength history
        self.history["S1"].append(self.associative_strengths["S1"])
        self.history["S2"].append(self.associative_strengths["S2"])

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