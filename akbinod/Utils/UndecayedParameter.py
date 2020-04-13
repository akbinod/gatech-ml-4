from akbinod import DecayedParameter

class UndecayedParameter(DecayedParameter):
	def __init__(self, value):
		'''Almost a joke, this just implements the interface and stays steady at the start value.'''
		super().__init__(value, value, 1)
	def decay_parameter(self):
		return self.start
	def __str__(self):
		return f"Constant at {self.start}"