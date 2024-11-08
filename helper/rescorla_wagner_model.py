class RescorlaWagnerModel:
    def __init__(self, entities, learning_rates, max_reward=1.0):
        """
        Initialize with entities (stimuli or states), their learning rates, and maximum reward.
        """
        self.learning_rates = learning_rates
        self.associations = {entity: 0.0 for entity in entities}
        self.max_reward = max_reward

    def update_strengths(self, present_stimuli, reward):
        """
        Update associative strengths for each entity based on Rescorla-Wagner rule.
        """
        # Calculate prediction (sum of associative strengths of presented stimuli)
        prediction = sum(self.associations[stimulus] for stimulus in present_stimuli)

        # Calculate prediction error
        prediction_error = reward - prediction

        # Update associative strength for each presented stimulus
        for stimulus in present_stimuli:
            delta = self.learning_rates[stimulus] * prediction_error
            self.associations[stimulus] += delta

    def get_associations(self):
        """Return current associative strengths for each entity."""
        return self.associations
