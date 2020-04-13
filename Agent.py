
from akbinod import Plot
from Learners.defs import Result
from Recorders.BaseRecorder import BaseRecorder
import numpy as np
np.random.seed(0)
import json
import time
import os

from enum import Enum, auto

class convergence_measure(Enum):
	not_yet				= auto()
	score_met 			= auto(),
	loss_converged 		= auto()

class Agent():
	"""An agent that uses the provided Reinforcement Learner for solving OpenAI Gym (like) problems."""

	def __init__(self, learner, *, target_moving_avg = 200, run_name = "", recorder = None):

		self.learner = learner
		self.target_moving_avg = target_moving_avg
		self.run_name = run_name
		self.recorder = None

		if (not recorder is None and issubclass(recorder.__class__, BaseRecorder)):
			self.recorder = recorder

		self.goal_met = False
		self.convergence_by = convergence_measure.not_yet
		self.last_moving_avg = 0

		self.run_scores = []

		self.train_results = []
		self.play_results = []
		self.moving_avg = []

		self.curr_results = self.train_results
		self.training_episodes = 0
		self.run_max_tries = 0
		self.episodes = 0

	def start_run(self, min_tries, max_tries, training = True):
		self.run_min_tries = min_tries
		self.run_max_tries = max_tries
		if training:
			self.train_results = []
			self.curr_results = self.train_results
		else:
			self.play_results = []
			self.curr_results = self.play_results

		self.run_start_time = time.time()
		self.moving_avg = []

		self.goal_met = False
		return

	def close_episode(self, result):
		record = False
		#stash the results
		if not result is None:
			self.curr_results.append(result)
			self.run_scores.append(result.score)
			self.last_moving_avg = sum(self.run_scores)/len(self.run_scores)
			self.moving_avg.append(round(self.last_moving_avg,1))

		# pruning, and convergence checks happen every min_runs
		if (self.learner.ep_count + 1) % self.run_min_tries == 0:
			#just keep the last min scores of the following
			self.run_scores = self.run_scores[len(self.run_scores)-self.run_min_tries:]
			self.moving_avg = self.moving_avg[len(self.moving_avg)-self.run_min_tries:]

			# check if we've met the avg score requirement
			#print( round(mvavg,1))
			if self.last_moving_avg > self.target_moving_avg:
				self.goal_met = True
				self.convergence_by = convergence_measure.score_met

				print(f"Goal met! Moving average of {self.last_moving_avg} achieved at episode {self.learner.ep_count}.")
			elif self.learner.loss_converged:
				# check the loss scores to see if they have stopped changing
				self.goal_met = True
				self.convergence_by = convergence_measure.loss_converged

		# at the last 10 episodes - record something
		record = self.learner.ep_count >= self.run_max_tries - 10

		return self.goal_met, (self.recorder if record else None)

	def analyze_settings(self):
		res = {}
		res["run_name"] = self.run_name
		res["time_stamp"] = time.ctime(self.run_start_time)
		t2 = time.gmtime(time.time() - self.run_start_time)
		res["time"] = time.strftime('%H:%M:%S', t2 )
		res["learner_params"] = self.learner.params.to_json()

		res["goal_met"] = self.goal_met
		res["convergence_by"] = str(self.convergence_by)
		res["last_mv_avg"] = self.last_moving_avg
		res["run_max_tries"] = self.run_max_tries
		res["actions_stats"] = self.learner.action_stats

		return res

	def analyze_training(self):
		batch = Result(*zip(*self.train_results))

		self.analysis = res = {}
		res["settings"] = self.analyze_settings()
		res["state_visits"] = self.learner.visits_tracker.tolist()
		res["training_episodes"] = self.training_episodes
		res["avg_loss"] = float(round(np.mean(batch.loss),2))
		res["std"] = float(round(np.std(batch.loss),2))
		res["loss"] = {}
		res["loss"]["max"] = int(np.max(batch.loss))
		res["loss"]["max_at_ep_step"] = (int(np.argmax(batch.loss)), batch.steps[int(np.argmax(batch.loss))])
		res["loss"]["min"] = int(np.min(batch.loss))
		res["loss"]["min_at_ep_step"] = (int(np.argmin(batch.loss)),  batch.steps[int(np.argmin(batch.loss))])
		res["score"] = self.analyze_rewards(batch.score,batch.steps)


		#print(json.dumps(res))
		#these will be shown after we finish play - workaround for the matplotlib freezing bug
		self.train_plot = Plot("Training Scores", "episodes","rewards",batch.score)
		self.epsilon_plot = Plot("Decayed Epsilon", "steps", "epsilon", self.learner.params.epsilon.usage)
		self.loss_plot = Plot("Training Loss", "episodes","loss",batch.loss)
		self.train_steps_plot = Plot("Training Steps", "episodes","ste[s]",batch.steps)
		self.convergence_plot = Plot("Covergence","Iteration","Error", self.learner.convergence_tracker)
		self.state_visits_plot = None
		t = self.learner.visits_tracker
		if not t is None:
			self.state_visits_plot = Plot("State Visits","state","frequency", t)

		return
	def analyze_rewards(self, rewards, steps):

		res = {}
		res["avg_score"] = round(np.mean(rewards),2)
		res["std"] = float(round(np.std(rewards),2))
		res[f"over_{self.target_moving_avg}"] = len([j for j in rewards if j >= self.target_moving_avg and j < 2 * self.target_moving_avg])
		res[f"over_{2 * self.target_moving_avg}"] = len([j for j in rewards if j >= 2 * self.target_moving_avg])
		res["max"] =  int(np.max(rewards))
		res["max_at_ep_step"] = (int(np.argmax(rewards)), steps[int(np.argmax(rewards))])
		res["min"] = int(np.min(rewards))
		res["min_at_ep_step"] = (int(np.argmin(rewards)), steps[int(np.argmin(rewards))])
		res["steps"] = {}
		res["steps"]["max"] = int(np.max(steps))
		res["steps"]["max_at_ep_rew"] = (int(np.argmax(steps)), rewards[int(np.argmax(steps))])
		res["steps"]["min"] = int(np.min(steps))
		res["steps"]["min_at_ep_rew"] = (int(np.argmin(steps)), rewards[int(np.argmin(steps))])

		res["scores"] = rewards
		res["moving_avg"] = self.moving_avg

		return res


	def analyze_play(self):
		batch = Result(*zip(*self.play_results))

		self.analysis = res = {}
		res["settings"] = self.analyze_settings()
		res["episodes"] = self.episodes
		res["score"] = self.analyze_rewards(batch.score, batch.steps)

		#print(json.dumps(res))
		self.play_plot = Plot("Inference Score", "episodes","rewards",batch.score )

		return

	def play(self, min_tries, max_tries):

		self.start_run(min_tries, max_tries, False)
		self.learner.play(self, max_tries)
		self.episodes = self.learner.ep_count
		self.analyze_play()
		self.serialize(self.learner.params.data_path + ".play")

		return self
	def train(self, min_tries, max_tries):

		self.start_run(min_tries, max_tries, True)
		self.learner.train(self, max_tries)
		self.training_episodes = self.learner.ep_count
		self.analyze_training()
		self.serialize(self.learner.params.data_path + ".train")

		return

	def serialize(self, path):
		with open(path + ".json","a+") as f:
			f.write(json.dumps(self.analysis))
			f.write("\n")
		return
