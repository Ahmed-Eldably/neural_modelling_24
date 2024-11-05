from abc_classical_conditioning_paradigm import ClassicalConditioningParadigm


class InhibitoryConditioningParadigm(ClassicalConditioningParadigm):

    def pre_training(self, pre_training_trials=0):
        """Define the pre-training phase, if applicable."""
        pass

    def training(self, training_trials=0):
        """
        Parameters:
        - training_trials (int): Number of training trials
        """
        for trial in range(training_trials):
            if trial % 2 == 0:
                # present S1 alone with a reward to build a positive association
                present_stimuli = ["S1"]
                reward = self.lambda_  # Reward is given
            else:
                # present S1 and S2 together with no reward to build inhibition
                present_stimuli = ["S1", "S2"]
                reward = 0.0  # No reward given

            self.update_associative_strength(present_stimuli, reward)


train_trials = 100

inhibitory_conditioning = InhibitoryConditioningParadigm(alpha1=0.1,
                                                         alpha2=0.1,
                                                         lambda_=1.0)
inhibitory_conditioning.run(training_trials=train_trials)
inhibitory_conditioning.plot_history("Inhibitory Conditioning Paradigm")