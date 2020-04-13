from akbinod import DecayedParameter

class LittmanAlpha(DecayedParameter):
	'''Designed for use like Littman prescribes in his paper on Markov games using Game Theory.
		Littman uses aplpha such that in the early stages
		(when things are uninitiated and there's a lot of exploration going on)
		learning from Error (actual minus observed) is maximized.
		Once the model has learnt a bit, and Error is falling, we want to keep
		learning from the error term, but also give more credence to what we have
		already learnt.
	'''
	def __init__(self,*, start = 1, min = 0.0001, anticipatedStepsT = 1000000, decay=0.9999954):
		super().__init__(start, min, anticipatedStepsT)
		self.decay_value = decay

	def decay_parameter(self):
		#  when step = 1, this resolves to the starting value
		 ret = self.start * (self.decay_value ** ((self.t - 1) if self.t > 0 else 0))

		 return ret

	def reset(self):
		# represents the step number that we increment on every call to next
		self.t = 0
		self.usage = []
