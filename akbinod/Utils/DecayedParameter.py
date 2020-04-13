from akbinod.Utils.Plotting import Plot

class DecayedParameter:
	""" Implements a decayed parameter (base class implements exponential decay)
	e.g., Lambda ala Sutton & Barto pg 290, figure 12.2
	Principally provides a standard interface for
	generic loops like Q learning etc.
	It tracks usage so that you can plot the values used across episodes etc.
	To implement a different type of decay, extend this class
	and override the following:
		1. __init__
		2. reset
		3. decay_parameter .
		4. __str__
	"""
	def __init__(self, start, min, anticipatedStepsT):
		self.start = start

		# represents the total steps anticipated in this episode. Might not always work out
		# to be just that, because you might be playing Lundar Lander or something like that
		# where the number of steps in an episode is not deterministic
		self.T = anticipatedStepsT
		# the minimum Lambda or Alpha that you want to use
		self.min = min

		self.__current = start
		# get things initialized
		self.reset()

	def __safe(self, p):
		if p > self.start:
			#happens when you guess the number of steps wrong
			p = self.min
		elif p < self.min:
			#well the dev wants a floor
			p = self.min
		return p

	def decay_parameter(self):
		'''Override this method to implement something other than an exponential decay'''
		return self.start ** (self.t)

	@property
	def current(self):
		ret = self.__safe(self.decay_parameter())
		self.usage.append(ret)
		return ret

	@property
	def next(self):
		#advances the time step
		self.t += 1

		return self.current

	def atN(self, n):
		return self.start ** (n-1)

	def reset(self):
		# represents the step number that we increment on every call to next
		self.t = 1
		self.usage = []

		return

	def __str__(self):
		return str((self.start, self.min, self.T))