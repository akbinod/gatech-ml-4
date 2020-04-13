from Learners.GridWorldSimpleQPolicy import GridWorldSimpleQPolicy

class FrozenLakePolicy(GridWorldSimpleQPolicy):
	def __init__(self, env_states, env_actions, learner_params, env):
		super().__init__(env_states, env_actions, learner_params, env)

	def next_action(self, *, state=None):
		ret = 0
		# don't know the action yet

		actionV = self.get_state_value(state, None)
		if sum(actionV) == 0:
			# tied at 0 all, select to go East since
			# at the beginning, frozen_lake's goal is
			# east of the starting point
			ret = 2
		else:
			ret = actionV.argmax()
		return ret