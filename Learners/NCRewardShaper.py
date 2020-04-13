from Learners.IRewardShaper import IRewardShaper
import numpy as np

class NCRewardShaper(IRewardShaper):
	def __init__(self, states):
		super().__init__()
		self.map = map
		# num of states
		self.width = states-1
		self.rewards = np.zeros((self.width), dtype=float)
		for i in range(1, self.width - 1):
			self.rewards[i] = -0.1

	def shape(self, learner, sars):

		reward =  sars.r #+ self.rewards[sars.s]
		if sars.sprime> sars.s:
			reward += 0.1
		elif sars.sprime < sars.s:
			reward -= 0.1

		return reward

	def __str__(self):

		return "Reward going in a rightward direction."