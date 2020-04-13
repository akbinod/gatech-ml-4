
# Purely abstract base class/Interface
class IPolicy():
	def serialize(self):
		raise NotImplementedError()
	def deserialize(self):
		raise NotImplementedError()
	def begin_training(self):
		raise NotImplementedError()
	def begin_inference(self):
		raise NotImplementedError()
	def next_action(self, **kwargs):
		raise NotImplementedError()
	def learn(self, sars, episode_done):
		raise NotImplementedError()

	@property
	def state_visit_tracker(self):
		return None