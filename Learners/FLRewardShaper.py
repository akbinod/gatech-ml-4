from Learners.IRewardShaper import IRewardShaper
import numpy as np

class FLRewardShaper(IRewardShaper):
	def __init__(self, map):
		super().__init__()
		self.map = map
		# num of states
		self.width = len(self.map[0])
		self.rewards = np.zeros((self.width, self.width), dtype=float)
		for r in range(self.width):
			for c in range(self.width):
				R= 0
				if self.map[r][c] == 'G':
					R = 1
				elif self.map[r][c] == 'H':
					R = -1
				else:
					R = -0.01
				self.rewards[r][c] = R

		self.rewards = np.reshape(self.rewards,(self.width*self.width))
	def shape(self, learner, sars):

		reward = self.rewards[sars.s]

		return reward

	def __str__(self):

		return "energy consumption tax of -0.01; -1 for falling into a hole"