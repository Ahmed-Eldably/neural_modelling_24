from abc_classical_conditioning_paradigm import ClassicalConditioningParadigm

class ExplainingAway(ClassicalConditioningParadigm):
    def pre_training(self, pre_training_trials=0):
        """
        Pre-training phase for explaining away

        Parameters:
        - pre_training_trials (int): Number of pre-training trials
        """

        # S1 and S2 are independently associated with the reward.
        for trial in range(pre_training_trials):
            if trial % 2 == 0:
                # Even trials: Present S1 alone with a reward
                present_stimuli = ["S1"]
                reward = self.lambda_
            else:
                # Odd trials: Present S2 alone with a reward
                present_stimuli = ["S2"]
                reward = self.lambda_

            self.update_associative_strength(present_stimuli, reward)

    def training(self, training_trials=0):
        """
        Training phase for explaining away

        Parameters:
        - training_trials (int): Number of training trials
        """
        # S1 and S2 are presented together with a reward
        for trial in range(training_trials):
            present_stimuli = ["S1", "S2"]
            reward = self.lambda_
            self.update_associative_strength(present_stimuli, reward)


pre_train_trials = 50
train_trials = 50

explaining_away = ExplainingAway(alpha1=0.1,
                                 alpha2=0.1,
                                 lambda_=1.0)
explaining_away.run(pre_training_trials=pre_train_trials,
                    training_trials=train_trials)
explaining_away.plot_history("Explaining Away Paradigm")