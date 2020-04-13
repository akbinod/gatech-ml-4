import numpy as np
np.random.seed(903549491)
from Learners.BasePolicy import BasePolicy
import random
random.seed(903549491)

import copy
import os

from akbinod import DecayedParameter, UndecayedParameter, TimedFunction
from gym.utils import colorize
import sys
from six import StringIO
from contextlib import closing
from Recorders import GymStringRecorder
from Learners.defs import _MAPS, _ACTIONS

# →
# RIGHTWARDS ARROW
# Unicode: U+2192, UTF-8: E2 86 92
# ←
# LEFTWARDS ARROW
# Unicode: U+2190, UTF-8: E2 86 90
# ↑
# UPWARDS ARROW
# Unicode: U+2191, UTF-8: E2 86 91
# ↓
# DOWNWARDS ARROW
# Unicode: U+2193, UTF-8: E2 86 93



class FrozenLake(BasePolicy):
	def __init__(self, params):
		#initialize our controls, and use the provided map
		self.map = _MAPS[params.map_key]
		# num of states
		self.width = len(self.map[0])

		super().__init__(self.width * self.width,len(_ACTIONS), params)
		self.convergence_tracker = []

		# always square, so you have "states" as the number of rows and columns
		# each action cell will show the value of that action for that row and col
		self.u_table = np.zeros((self.width, self.width))
		#its assumed that outcomes are laid out as 0.8 - expected, 0.1 rotated left, 0.1 rotated right
		self.outcomes = 3
		# each outcome cell will contain a tuple showing:
		# 	0. the row you will end up in,
		# 	1. the column you will end up in,
		self.t_matrix = np.zeros((self.width, self.width,  self.env_actions, self.outcomes), dtype=object)

		# The policy matrix will be updated at the end after convergence
		# The absorbing states will be left as is, and each of the
		# other cells will be marked with the action to take therein.
		self.policy = np.empty((self.width, self.width),dtype=object)

		if self.model is not None:
			# something got rehydrated
			# check that the dimensions are the same
			if self.model.shape != self.policy.shape:
				self.model = None
			else:
				self.benchmark = copy.deepcopy(self.model)

		if self.model is None:
			self.model = np.zeros((self.width, self.width),dtype=int)

		# The reward matrix holds:
		# 	0. the cell's R
		# 	1. Whether it is a terminating cell
		self.rewards = np.empty_like(self.policy,dtype=object)

		for r in range(self.width):
			for c in range(self.width):
				R = 0
				term = False
				# initialize the policy with holes and goals
				if self.map[r][c] == 'G':
					self.policy[r][c] = colorize("o", 'green', highlight=True)
					R = 1
					term = True
				elif self.map[r][c] == 'H':
					self.policy[r][c] = colorize("o", 'red', highlight=True)
					R = -1
					term = True
				else:
					self.policy[r][c] = ''
				self.rewards[r][c] = (R, term)

				# build the transition matrix
				# at each cell, if we take some action, where are we likely to end up
				for a in range(self.env_actions):
					norm, left, right = self.get_next_states(r,c,a)
					self.t_matrix[r][c][a][0] = norm
					self.t_matrix[r][c][a][1] = left
					self.t_matrix[r][c][a][2] = right



	def get_next_states(self, r,c, a):
		'''Builds the transition map'''
		r_prime = r_left = r_right = r
		c_prime = c_left = c_right = c

		# follows from the actions map above
		# straight case
		if a == 0:
			c_prime -= 1
			r_left 	+= 1
			r_right -= 1
		elif a == 1:
			r_prime += 1
			c_left 	+= 1
			c_right -= 1
		elif a == 2:
			c_prime += 1
			r_left -= 1
			r_right += 1
		else:
			r_prime -= 1
			c_left 	-= 1
			c_right += 1

		r_prime = self.normalize(r_prime)
		c_prime = self.normalize(c_prime)

		r_left = self.normalize(r_left)
		c_left = self.normalize(c_left)

		r_right = self.normalize(r_right)
		c_right = self.normalize(c_right)

		return (r_prime, c_prime), (r_left, c_left), (r_right, c_right)

	def normalize(self, any):
		if any < 0:
			any = 0
		elif any >= self.width:
			any = self.width - 1
		return any

	@TimedFunction(True)
	def vi(self):
		self.render_process = "Value Iteration"
		self.eval_iterations = 0
		self.improve_iterations = 0
		self.model = np.zeros((self.width, self.width),dtype=int)
		self.visualize_policy()

		# clear out whatever we may be holding by way of error information
		self.convergence_tracker = []
		new_u_table = copy.deepcopy(self.u_table)
		rec = GymStringRecorder(self.params.data_path,self)
		rec.begin(0)
		done = False
		for self.eval_iterations in range(self.params.max_iterations):
			max_error = 0
			for r in range(self.width):
				for c in range(self.width):
					# default reward
					R = self.rewards[r][c][0]
					term_state = self.rewards[r][c][1]
					# by definition, in a terminating state,
					# the next action's value is 0 - you are'nt
					# going anywhere from here
					oa_utility = 0
					if not term_state:
						# regular skatable state - calculate
						optimal_action, oa_utility, oa_index = self.get_next_action(r,c)
						self.model[r][c] = oa_index

					new_utility = R + (self.params.gamma * oa_utility)
					new_u_table[r][c] = new_utility

					old_utility = self.u_table[r][c]
					error = abs(new_utility - old_utility)
					if error > max_error:
						max_error = error

			# done looping all cells
			# update the utility table
			self.u_table = copy.deepcopy(new_u_table)
			self.convergence_tracker.append(max_error)
			self.visualize_policy()
			# print the policy as it stands
			rec.capture_frame()

			# convergence check and early termination
			# out of the max_iterations loop
			if max_error < self.params.theta * ((1 - self.params.gamma)/self.params.gamma):
				# we are converged
				done = True
				self.serialize()
				break

		rec.end(0,self.eval_iterations)

	def get_next_action(self, r, c):
		optimal_action = ""
		optimal_action_utility = 0.0
		nextVSAs = []
		for a in range(self.env_actions):
			# these are the places you are likely to end up in with 0.8, 0.1, 0.1 prob
			norm_pos = self.t_matrix[r,c,a,0]
			left_pos = self.t_matrix[r,c,a,1]
			right_pos = self.t_matrix[r,c,a,2]

			# at each row, col, action we have the value
			# what happens if we manage to go the direction we want
			norm_val = 	self.u_table[norm_pos[0]][norm_pos[1]]
			# or get rotated left
			left_val = 	self.u_table[left_pos[0]][left_pos[1]]
			# or get rotated right
			right_val = self.u_table[right_pos[0]][right_pos[1]]

			# the value of taking action (a) in next state (r,c)
			new_vsa = (0.8 * norm_val) + (0.1 * left_val) + (0.1 * right_val)
			nextVSAs.append(new_vsa)

		optimal_action_utility = max(nextVSAs)
		optimal_action_index = np.argmax(nextVSAs)
		optimal_action = _actions[optimal_action_index]

		return optimal_action, optimal_action_utility, optimal_action_index

	def get_action_utility(self, r, c, action):
		action_utility = 0.0
		nextVSAs = []

		# these are the places you are likely to end up in with 0.8, 0.1, 0.1 prob
		norm_pos = self.t_matrix[r,c,action,0]
		left_pos = self.t_matrix[r,c,action,1]
		right_pos = self.t_matrix[r,c,action,2]

		# at each row, col, action we have the value
		# what happens if we manage to go the direction we want
		norm_val = 	self.u_table[norm_pos[0]][norm_pos[1]]
		# or get rotated left
		left_val = 	self.u_table[left_pos[0]][left_pos[1]]
		# or get rotated right
		right_val = self.u_table[right_pos[0]][right_pos[1]]

		# the value of taking action (a) in next state (r,c)
		action_utility = (0.8 * norm_val) + (0.1 * left_val) + (0.1 * right_val)

		return action_utility

	def dump_policy(self):
		# print the policy as it stands
		print('\n')
		for r in range(self.width):
			row = ""
			for c in range(self.width):
				row += ' ' + self.policy[r][c]
			print(row)
		print('\n')

	@TimedFunction(True)
	def abandoned_pi(self):
		# clear out whatever we may be holding by way of error information
		self.convergence_tracker = []
		# # randomly create a policy
		# l = random.choices(population=np.arange(self.width),k=self.width*self.width)
		# proposed_policy = np.array(l)
		# # # have the random policy reshaped to look
		# # # like our utility matrix
		# # proposed_policy = np.reshape(proposed_policy,(self.width, self.width))
		# new_u_table = np.zeros((3, (self.width*self.width)),dtype=float)
		# t_matrix = self.t_matrix.reshape(self.width * self.width, self.env_actions, self.outcomes)
		# done = False
		# for i in range(self.params.max_iterations):
		# 	for j in range(len(proposed_policy)):
		# 		this_a = proposed_policy[j]
		# 		for o in range(self.outcomes):
		# 			pos = t_matrix[j][this_a][o]
		# 			# new_u_table[j][o] =

		# 	for i in range(self.width * self.width):
		# 		pols[i] = np.random.RandomState.randint(0,self.env_actions)

		# 	for r in range(self.width):
		# 		for c in range(self.width):




		# 	max_error = 0
		# 	for r in range(self.width):
		# 		for c in range(self.width):
		# 			# default reward
		# 			R = self.rewards[r][c][0]
		# 			term_state = self.rewards[r][c][1]
		# 			# by definition, in a terminating state,
		# 			# the next action's value is 0 - you are'nt
		# 			# going anywhere from here
		# 			oa_utility = 0
		# 			if not term_state:
		# 				# regular skatable state - calculate
		# 				optimal_action, oa_utility = self.get_next_action(r,c)
		# 				self.policy[r][c] = optimal_action

		# 			new_utility = R + (self.params.gamma * oa_utility)
		# 			new_u_table[r][c] = new_utility

		# 			old_utility = self.u_table[r][c]
		# 			error = abs(new_utility - old_utility)
		# 			if error > max_error:
		# 				max_error = error
		# 	# done looping all cells
		# 	# update the utility table
		# 	self.u_table = copy.deepcopy(new_u_table)
		# 	self.convergence_tracker.append(max_error)
		# 	# convergence check and early termination
		# 	# out of the max_iterations loop
		# 	# max_error = max(episode_errors)
		# 	# if i >= (self.params.max_iterations / 10) and max_error < self.params.theta:
		# 	if max_error < self.params.theta * ((1 - self.params.gamma)/self.params.gamma):
		# 		# we are converged
		# 		done = True
		# 		print(f"Converged at iteration {i}")

		# 		break

		# # print the policy as it stands
		# self.dump_policy()

	@TimedFunction(True)
	def pi(self):
		self.render_process = "Policy Iteration"
		self.eval_iterations = 0
		self.improve_iterations = 0
		self.model = np.zeros((self.width, self.width),dtype=int)

		def eval_pol():

			# new_u_table = copy.deepcopy(self.u_table)
			for self.eval_iterations in range(self.params.max_iterations):
				max_error = 0
				for r in range(self.width):
					for c in range(self.width):
						# default reward
						R = self.rewards[r][c][0]
						term_state = self.rewards[r][c][1]
						# by definition, in a terminating state,
						# the next action's value is 0 - you are'nt
						# going anywhere from here
						action_utility = 0
						if not term_state:
							# regular skatable state - calculate
							# lets go with our randomly proposed action
							this_action = self.model[r][c]
							action_utility = self.get_action_utility(r,c, this_action)

						new_utility = R + (self.params.gamma * action_utility)
						new_u_table[r][c] = new_utility

						old_utility = self.u_table[r][c]
						error = abs(new_utility - old_utility)
						if error > max_error:
							max_error = error
				# done looping all cells
				# update the utility table
				self.u_table = copy.deepcopy(new_u_table)

				# convergence check and early termination
				# out of the max_iterations loop
				# if i >= (self.params.max_iterations / 10) and max_error < self.params.theta:
				if max_error < self.params.theta * ((1 - self.params.gamma)/self.params.gamma):
					# we are converged
					done = True
					break

		def improve_pol():
			max_error = 0
			done = True
			for r in range(self.width):
				for c in range(self.width):
					curr_policy = self.model[r][c]
					guessed_utility = self.u_table[r][c]
					# default reward
					R = self.rewards[r][c][0]
					term_state = self.rewards[r][c][1]
					# by definition, in a terminating state,
					# the next action's value is 0 - you are'nt
					# going anywhere from here
					oa_utility = 0
					if not term_state:
						# regular skatable state - calculate
						optimal_action, oa_utility, oa_index = self.get_next_action(r,c)
						if curr_policy != oa_index:
							# switch it out
							self.model[r][c] = oa_index
							done = False

					error = abs(oa_utility - guessed_utility)
					if error > max_error:
						max_error = error

			# policy error per Russel, Norvig
			self.convergence_tracker.append(max_error)
			return done

		# clear out whatever we may be holding by way of error information
		self.convergence_tracker = []
		new_u_table = copy.deepcopy(self.u_table)

		# set up a random policy
		for r in range(self.width):
			l = random.choices(population=np.arange(self.env_actions),k=self.width)
			for c in range(self.width):
				self.model[r][c] = l[c]
		self.visualize_policy()

		# set up the recorder only after the random policy
		# has been set up
		recorder = GymStringRecorder(self.params.data_path,self)
		# do the loop of evaluating and then improving ad infinitum
		done = False
		recorder.begin(0)
		while not done:
			self.improve_iterations += 1
			eval_pol()
			done = improve_pol()
			self.visualize_policy()
			# grab the latest result
			recorder.capture_frame()
			# self.render("Policy Iteration",eval_iterations + 1, improve_iterations + 1,'human')
		recorder.end(0,self.improve_iterations)
		if done:
			self.serialize()

	def visualize_policy(self):
		# convert the action indexes to policy actions
		for r in range(self.width):
			for c in range(self.width):
				if not self.rewards[r][c][1]:
					# if we are not in a terminal state
					ch = _actions[self.model[r][c]]
					if self.benchmark is not None:
						# we have a benchmark
						if self.model[r][c] != self.benchmark[r][c]:
							# and it do3s not match what we are using
							ch = colorize(ch, "gray", highlight=True)
					self.policy[r][c] = ch

	def render(self, mode='ansi', close=False):
		ret = None
		out = np.asarray(self.policy.copy().tolist())
		# don't need the next line - we don't have an array of strings
		# out = [[c.decode('utf-8') for c in line] for line in out]

		# write out the policy
		outfile = StringIO() if mode == 'ansi' else sys.stdout
		outfile.write(f"{self.render_process}\n")
		outfile.write("\n ")
		outfile.write("\n ".join([" ".join(row) for row in out]) + " \n")
		outfile.write("\n")
		# sundry stats
		outfile.write(f"Eval Iter: {self.eval_iterations}\n")
		outfile.write(f"Impr Iter: {self.improve_iterations}\n")
		outfile.write("\n")

		# No need to return anything for human
		if mode != 'human':
			with closing(outfile):
				ret = outfile.getvalue()

		return ret

	@property
	def state(self):
		return self.model

	@state.setter
	def state(self,val):
		self.model = val