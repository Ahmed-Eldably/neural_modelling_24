from abc_classical_conditioning_paradigm import ClassicalConditioningParadigm


class Overshadowing(ClassicalConditioningParadigm):
    def pre_training(self, pre_training_trials=0):
        pass

    def training(self, training_trials=0):
        """
        The training phase for overshadowing

        - training_trials (int): Number of training trials
        """
        # both S1 and S2 are presented together with a reward
        for trial in range(training_trials):
            present_stimuli = ["S1", "S2"]
            reward = self.lambda_
            self.update_associative_strength(present_stimuli, reward)

training_trials = 100
pre_training_trials = 50
overshadowing = Overshadowing(alpha1=0.1,
                              alpha2=0.05,
                              lambda_=1.0)
overshadowing.run(pre_training_trials=pre_training_trials,
                  training_trials=training_trials)
overshadowing.plot_history("Overshadowing Paradigm")
