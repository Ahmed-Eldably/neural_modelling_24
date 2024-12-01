from abc_classical_conditioning_paradigm import ClassicalConditioningParadigm


class SecondaryConditioning(ClassicalConditioningParadigm):
    def pre_training(self, pre_training_trials=0):
        """
        Pre-training phase for Secondary Conditioning

        Parameters:
        - pre_training_trials (int): Number of pre-training trials
        """
        # S1 is presented alone with a reward
        for trial in range(pre_training_trials):
            present_stimuli = ["S1"]
            reward = self.max_reward
            self.update_associative_strength(present_stimuli, reward)

    def training(self, training_trials=0):
        """
        Training phase for Secondary Conditioning

        Parameters:
        - training_trials (int): Number of training trials
        """
        # both S1 and S2 are presented together with a reward.
        for trial in range(training_trials):
            present_stimuli = ["S1", "S2"]
            reward = 0.0
            self.update_associative_strength(present_stimuli, reward)


pre_train_trials = 10
train_trials = 90

secondary_conditioning = SecondaryConditioning(learning_rate_1=0.1,
                                               learning_rate_2=0.1,
                                               max_reward=1.0)
secondary_conditioning.run(pre_training_trials=pre_train_trials,
                           training_trials=train_trials)

secondary_conditioning.plot_history("Secondary Conditioning Paradigm")