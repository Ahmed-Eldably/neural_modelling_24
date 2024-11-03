
from assignment_02.classical_conditioning_paradigm import ClassicalConditioningParadigm

class InhibitoryConditioningParadigm(ClassicalConditioningParadigm):

    def start_pre_training(self, trials):
        """Pre-Training Phase: Only S1 is paired with reward."""
        for _ in range(trials):
            self._update(present_stimuli=["S1"], reward=1)
            self.history["S2"].append(self.V["S2"])

    def start_training(self, trials):
        """Training Phase: Both S1 and S2 are presented with no reward."""
        for _ in range(trials):
            self._update(present_stimuli=["S1", "S2"], reward=0)



# Initialize and run the inhibitory conditioning simulation
inhibitory_paradigm = InhibitoryConditioningParadigm(alpha=0.1, lambda_=1.0)

# Define number of trials for each phase
trials_pre_training = 10
trials_training = 10

inhibitory_paradigm.start_pre_training(trials=trials_pre_training)
inhibitory_paradigm.start_training(trials=trials_training)
inhibitory_paradigm.plot_history(title="Inhibitory Conditioning in Rescorla-Wagner Model")
