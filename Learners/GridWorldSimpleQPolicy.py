from Learners.BasePolicy import BasePolicy
from akbinod import LittmanAlpha
import numpy as np
np.random.seed(903549491)
import random
random.seed(903549491)

class GridWorldSimpleQPolicy(BasePolicy):
	def __init__(self, env_states, env_actions, learner_params, env):
		self.QTable = None
		super().__init__(env_states, env_actions, learner_params)

		# self.rows = env.nrow
		# self.columns = env.ncol

		if self.QTable is None:
			# nothing was rehydrated from disk
			# get a new one going
			self.begin_training()

	def learn(self, sars, episode_done):
		''' Update the value of the current Q(s,a) '''
		# step the alpha
		alpha = self.params.alpha.next
		# must use the action that we went with, not the argmax
		thisQSA = self.get_state_value(sars.s, sars.a)
		# this is the q(sprime, aprime) when state is terminal
		nextQSA = 0.
		if not episode_done:
			# calc next qsa
			nextAction = self.next_action(state=sars.sprime)
			nextQSA = self.get_state_value(sars.sprime,nextAction)

		target = sars.r + (self.params.gamma * nextQSA)
		error = target - thisQSA
		# prepare the update
		update = ((1 - alpha) * thisQSA) + (alpha * target)
		self.set_state_value(sars.s, sars.a, update)

		return error


	def next_action(self, *, state=None):
		ret = 0
		# don't know the action yet

		actionV = self.get_state_value(state, None)
		if sum(actionV) == 0:
			# tie-breaker returns one at random
			ret = np.random.randint(0 ,self.env_actions)
		else:
			ret = actionV.argmax()
		return ret

	def begin_training(self):
		# initialize QTable

		# (rows X columns), and actions
		self.QTable = np.zeros((self.env_states, self.env_actions))

		# to track all the states we have visited
		self.visited_states = np.zeros((self.env_states), dtype=int)

	def begin_inference(self):
		# nothing special for us to do
		pass

	@property
	def state(self):
		return self.QTable

	@state.setter
	def state(self,val):
		self.QTable = val

	def get_state_value(self, state, action):
		ret = None
		if action is None:
			ret = self.QTable[state, :]
		else:
			ret = self.QTable[state, action]
		return ret

	def set_state_value(self, state, action, val):
		self.QTable[state, action] = val
		# we've visited this state, so increment the counter
		self.visited_states[state] += 1

		return
	@property
	def state_visit_tracker(self):
		return self.visited_states

