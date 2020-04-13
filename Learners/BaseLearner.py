from collections import namedtuple
from Learners.BasePolicy import BasePolicy
from Recorders.BaseRecorder import BaseRecorder
from Learners.defs import Result, Sars
from akbinod.Utils import TimedFunction
import akbinod


import numpy as np
np.random.seed(903549491)
import gym
import json
import os


#Base class for all learners (abstract)
class BaseLearner():
	def __init__(self, env, learner_params):
		#this is our gym (like) environment that we are solving for
		self.env = env
		self.params = learner_params

		try:
			self.env_states = env.observation_space.shape[0]
		except:
			self.env_states = env.observation_space.n

		self.env_actions = env.action_space.n

		self.train_actions = np.zeros(self.env_actions).tolist()
		self.play_actions = np.zeros(self.env_actions).tolist()
		self.actions_buffer = self.play_actions
		self.rand_v_optimal_split = [0,0]

		# There has to be a policy, and this will be
		# set by the deriver
		self.policy = None
		# get everything initialized - assume training is going to happen
		self.start_run(True)
		self.avg_loss = None

	@property
	def policy(self):
		return self.__policy

	@policy.setter
	def policy(self,val):
		if val is None:
			self.__policy = None
		else:
			if issubclass(val.__class__, BasePolicy):
				self.__policy = val
			else:
				raise ValueError(val)
		return

	def start_run(self, training):
		#reset everything
		self.state = self.env.reset()
		self.ep_count = 0
		self.ep_steps = 0
		self.ep_reward = 0.0
		self.ep_loss = 0.0
		self._convergence_tracker = []
		if training:
			self.train_actions = np.zeros(self.env_actions).tolist()
			self.actions_buffer = self.train_actions
		else:
			self.play_actions = np.zeros(self.env_actions).tolist()
			self.actions_buffer = self.play_actions
			#sanity checks here
			#do we have a good model file?
			self.params.check_model_file_exists()
		return

	def close_episode(self, agent):
		self.state = self.env.reset()
		res = Result(self.ep_steps, int(self.ep_reward), int(self.ep_loss))
		goal_met, record = agent.close_episode(res)

		self.ep_count += 1
		self.ep_steps = 0
		self.ep_reward = 0.0
		self.ep_loss = 0.0

		return goal_met, res, record

	@property
	def loss_converged(self):
		'''The last 10 mean loss values must be within theta of each other for this to return true.'''
		ct = self.convergence_tracker
		ll = len(ct)
		means = []
		if ll > 10:
			# if the average loss of the last 10 loss scores
			# is within theta of each other, then we're converged
			for i in range(10):
				means.append(np.mean(ct[0 :ll-i]))

			done = True
			for i in range(len(means)-1):
				if abs(means[i] - means[i+1]) > self.params.convergence_theta:
					done = False
					break


	@property
	def action_stats(self):
		ret = {}
		the_split = {}
		the_split["total_steps"] = sum(self.actions_buffer)
		asplit = [round(num/sum(self.actions_buffer),2) for _, num in enumerate(self.actions_buffer)]
		if not self.params.action_map is None:
			match = [1 for i,j in enumerate(asplit) if str(i) in self.params.action_map.keys()]
			if len(match) == len(asplit) == len(self.params.action_map.keys()):
				for i,j in enumerate(asplit):
					the_split[self.params.action_map[f"{i}"]] = j
			else:
				the_split["error"] = "Provided map does not satisfy tests. Raw split included."
				the_split["raw"] = asplit
		else:
			the_split["error"] = "No map provided. Raw split included."
			the_split["raw"] = asplit

		ret["actions_split"] = the_split
		ret["rand_v_optimal"] = [round(num/sum(self.rand_v_optimal_split),2) for _, num in enumerate(self.rand_v_optimal_split)]
		return ret

	def step(self, training_mode = True):
		#increment the step number here
		self.ep_steps += 1

		if training_mode and (np.random.random() <= self.params.epsilon.next):
			#try something random if we are in training and
			#  we meet the epsilon test
			# sample from the environment
			action = np.random.randint(0, self.env_actions)
			self.rand_v_optimal_split[0] += 1
		else:
			#get the action from the deriving class
			# We always go greedy when in inference mode, because
			# exploration is a function in the training mode only.
			action = self.policy.next_action(state = self.state)
			self.rand_v_optimal_split[1] += 1
		#track the actions distr, so we can spot
		# issues like a skewed distribution
		self.actions_buffer[action] += 1
		s = self.state

		#step in the environment
		self.state , reward, done, _  = self.env.step(action)

		#acrete to episode rewards
		self.ep_reward += reward
		sars = Sars(s,action,reward,self.state)
		return sars, done

	def train(self, agent, max_tries):
		self.start_run(True)

		self.params.training_episodes = max_tries
		self.params.epsilon.T = max_tries
		self.params.epsilon.reset()
		#let the policy object do whatever it needs
		#to init its values
		self.policy.begin_training()

		# report = issubclass(self.env.env.__class__, akbinod.BaseEnv)
		report = False
		recorder = None
		#going to decay epsilon across episodes, not within
		while self.ep_count < max_tries:
			# starting a new episode here
			episode_done = 0
			if recorder: recorder.begin(self.ep_count +1)
			while not episode_done:
				#step the environment
				sars , episode_done = self.step(training_mode=True)
				# we're training, so discount by gamma
				sars = Sars(sars.s, sars.a, sars.r * self.params.gamma, sars.sprime)

				# Shape rewards if need be
				# reward shaping is done only during training,
				# and is used by the learner (or its policy)
				# to improve learning
				for sh in self.params.reward_shapers:
					r = sh.shape(self, sars)
					sars = Sars(sars.s, sars.a, r,sars.sprime)

				if report: self.env.env.agent_shaped_reward = sars.r
				if recorder: recorder.capture_frame()

				#learn something from this step if the subclass wants to
				loss = self.learn(sars, episode_done)
				self.ep_loss += loss
				if self.ep_steps == 1:
					# track this loss separately
					# only the loss from the very first step of an
					# episode is of value as this shows that in
					# progressive episodes, loss is coming down
					# i.e., Q values are converging
					self._convergence_tracker.append(loss)

			#done with an episode
			goal_met, result, rec  = self.close_episode(agent)
			if report: self.env.env.agent_score = result.score
			if recorder: recorder.end(result.score, result.steps)
			recorder = rec
			#decay the epsilon for the next episode
			self.params.epsilon.next

			if goal_met:
				self.policy.trained = True
				break

		# Done with our training loop, the policy
		# (e.g., weights in a neural network)
		# may or may not be trained at this point.
		# Save the policy , but do not overwrite
		# any existing policy unless this has met spec
		self.policy.serialize(goal_met)
		return

	def play(self, agent, max_tries):
		self.start_run(False)
		# put the policy into inference mode
		self.policy.begin_inference()
		report = issubclass(self.env.env.__class__, akbinod.BaseEnv)
		recorder = None
		for i in range(max_tries):
			#start of a new episode
			episode_done = 0
			if recorder: recorder.begin(i+1)
			while not episode_done:
				sars, episode_done = self.step(training_mode=False)
				if report: self.env.env.agent_shaped_reward = sars.r
				if recorder: recorder.capture_frame()

			#done with this particular episode
			goal_met, result, rec = self.close_episode(agent)
			if report: self.env.env.agent_score = result.score
			if recorder: recorder.end(result.score, result.steps)
			recorder = rec
			if self.params.always_learning: self.learn()
			#we've met our performance metric, get out
			if goal_met: break

		#done with all the episodes in play, or goal met
		return

	def learn(self, sars, done):
		raise NotImplementedError("Must override, even if you just 'pass'")

	@property
	def convergence_tracker(self):
		return self._convergence_tracker

	@property
	def visits_tracker(self):
		return self.policy.state_visit_tracker

