from Learners.IPolicy import IPolicy
from Learners.defs import Sars
import os
import json
import pickle

class BasePolicy(IPolicy):
	def __init__(self, env_states, env_actions, learner_params):
		#call the base class constructor
		super().__init__()

		self.env_states = env_states
		self.env_actions = env_actions
		self.params = learner_params
		self.trained = False
		#a parameters file might already exist
		#try loading it up - we might have pre-trained weights
		self.deserialize()

	@property
	def state(self):
		raise NotImplementedError()

	@state.setter
	def state(self,val):
		raise NotImplementedError()
	def serialize(self, overwrite=True):
		#let exceptions propagate
		nm = self.params.data_path + ".params"
		if not overwrite and os.path.exists(nm):
			# need a safe name
			nm = self.params.data_path
			for i in range(10000):
				nm = self.params.data_path + str(i) + ".params"
				if not os.path.exists(nm):
					break

		if not self.state is None:
			with open(nm, "wb+") as f:
				pickle.dump(self.state,f)
		return True

	def deserialize(self):
		#let exceptions propagate
		if os.path.exists(self.params.data_path + ".params"):
			with open(self.params.data_path + ".params", "rb") as f:
				self.state = pickle.load(f)
			# now that we've loaded something, we can claim to be all trained up
			self.trained = True
		return self.trained

	def begin_training(self):
		raise NotImplementedError()

	def begin_inference(self):
		raise NotImplementedError()

	def next_action(self, *, state = None):
		# Return the greedy action - the learner can figure
		# out the random action by itself.
		# Figure out the Qs
		# we're returning the action corresponding to whatever q_s_a is biggest
		# QsT = self(sT)
		# action = QsT.argmax().item()
		raise NotImplementedError
		return

	def learn(self, sars, episode_done):
		raise NotImplementedError()
