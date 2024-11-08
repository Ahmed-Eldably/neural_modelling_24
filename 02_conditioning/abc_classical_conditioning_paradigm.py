from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from helper.rescorla_wagner_model import RescorlaWagnerModel


class ClassicalConditioningParadigm(ABC):
    def __init__(self,
                 learning_rate_1=0.1,
                 learning_rate_2=0.1,
                 max_reward=1.0):
        """
        Parameters:
        - learning_rate_1 (float): Learning rate for stimulus S1
        - learning_rate_2 (float): Learning rate for stimulus S2
        - max_reward (float): Maximum associative strength (target reward value)
        """
        self.max_reward = max_reward
        self.history = {"S1": [],
                        "S2": []}

        self.rw_model = RescorlaWagnerModel(
            entities=["S1", "S2"],
            learning_rates={"S1": learning_rate_1, "S2": learning_rate_2}
        )

    def update_associative_strength(self, present_stimuli, reward, present_entities=None, belief=None):
        """
        Applies the Rescorla-Wagner rule to update associative strengths for each stimulus.

        Parameters:
        - present_stimuli (list): List of stimuli presented in the trial (for now we only have S1 and S2)
        - reward (float): The reward value for the trial
        """
        # Use Rescorla-Wagner model to update associative strengths
        self.rw_model.update_strengths(reward=reward, present_stimuli=present_stimuli)

        # Record associative strengths for each stimulus
        associations = self.rw_model.get_associations()
        for stimulus in associations.keys():
            self.history[stimulus].append(associations[stimulus])

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