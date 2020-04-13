

class IRewardShaper():
	'''Abstract base class for reward shaping'''
	def shape(self, learner, sars):
		'''Inspect sars and return a modified (or unmodified reward) '''
		raise NotImplementedError()

	def __str__(self):
		'''Tell us a little about what you do for the agent.'''
		raise NotImplementedError()