from classical_conditioning_paradigm import ClassicalConditioningParadigm


class BlockingParadigm(ClassicalConditioningParadigm):

    def start_pre_training(self, trials):
        """Pre-Training Phase: Only S1 is paired with reward."""
        for _ in range(trials):
            self._update(present_stimuli=["S1"], reward=1)
            self.history["S2"].append(self.V["S2"])

    def start_training(self, trials):
        """Training Phase: Both S1 and S2 are presented with reward."""
        for _ in range(trials):
            self._update(present_stimuli=["S1", "S2"], reward=1)


# Initialize and run the blocking paradigm simulation
blocking_paradigm = BlockingParadigm(alpha=0.1, lambda_=1.0)

# Define number of trials for each phase
trials_pre_training = 50
trials_training = 50

blocking_paradigm.start_pre_training(trials=trials_pre_training)
blocking_paradigm.start_training(trials=trials_training)
blocking_paradigm.plot_history("Blocking in Rescorla-Wagner Model")
