from Learners.BaseLearner import BaseLearner
from Learners.GridWorldSimpleQPolicy import GridWorldSimpleQPolicy

import gym

class NChainLearner(BaseLearner):
	def __init__(self, env, learner_params):
		super().__init__(env, learner_params)

		self.policy = GridWorldSimpleQPolicy(self.env_states, self.env_actions, self.params, self.env)

	def learn(self, sars, done):
		return self.policy.learn(sars,done)

	def step(self, training_mode = True):
		sars, done = super().step(training_mode)
		# if sars.sprime == (self.env_states - 1) or sars.sprime == 0:
		# 	# Fixing a bug in the nChain world
		# 	# when you get to the final cell, reset
		# 	# same thing for when you go past the
		# 	# beginning of the chain.
		# 	done = True
		return sars, done



