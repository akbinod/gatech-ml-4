from Learners.BaseLearner import BaseLearner
from Learners.GridWorldSimpleQPolicy import GridWorldSimpleQPolicy
from Learners.FrozenLakePolicy import FrozenLakePolicy

import gym

class FrozenLakeLearner(BaseLearner):
	def __init__(self, env, learner_params):
		super().__init__(env, learner_params)

		self.policy = FrozenLakePolicy(self.env_states, self.env_actions, self.params, self.env)

	def learn(self, sars, done):
		return self.policy.learn(sars,done)




