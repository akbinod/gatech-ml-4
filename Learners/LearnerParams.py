import json
import os
from akbinod import DecayedParameter

class LearnerParams():
	def __init__(self, env_name, *, epsilon = None, alpha = 0.0001, gamma = 0.99, target_refresh_rate_C = 500, mini_batch_size = 16, data_path = "./data", video_root = "./recordings", reward_shaper=None):

		self.env_name = env_name
		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma

		self.target_refresh_rate_C = target_refresh_rate_C
		self.mini_batch_size = mini_batch_size
		self.data_path = data_path
		self.video_path = video_root
		self.reward_shapers = []
		if not reward_shaper is None:
			self.reward_shapers.append(reward_shaper)
		#this may get filled in later
		self.action_map = None


		if not self.epsilon is None:
			if not issubclass(self.epsilon.__class__, DecayedParameter):
				#send me DecayedParameter object, or nothing at all
				raise TypeError()
		else:
			# if we were defaulted on the exploration rate, set something up
			self.epsilon = DecayedParameter(.995, .0001, 2000)

		# validate and adjust the path - do this after all the other params have
		# been taken care of, some of them are used in this
		#the path provided might be a dir, or a file (when there's a trained model already)
		if not os.path.exists(self.data_path):
			#we might be in a "play only" run, and this might
			# be the file name without the extension
			self.check_model_file_exists()
		else:
			#if this is just the directory name, then infre the file name
			if os.path.isdir(self.data_path):
				fname = self.infer_file_name()
				self.data_path = os.path.join(self.data_path, fname)
		if self.video_path != "":
			#if the user does not want video, thats fine, but supplied paths have to be directories
			if not os.path.exists(self.video_path) or not os.path.isdir(self.video_path):
				raise Exception(f"The following must be a folder in which to place recording files, and must exist: \n{self.video_path}")
			else:
				# at this point there is a proper file name for the data_path - use that
				ff = os.path.split(self.data_path)
				fname = ff[1]
				self.video_path = os.path.join(self.video_path, fname)


	def infer_file_name(self):
		fname = f"""{self.env_name}_gamma_{str(self.gamma).replace(".","_")}_eps_{str(self.epsilon).replace(".","_")}_alpha_{str(self.alpha).replace(".","_")}"""
		fname = fname.replace("(", "_")
		fname = fname.replace(")", "_")
		return fname

	def add_reward_shaper(self, reward_shaper):
		self.reward_shapers.append(reward_shaper)

	def check_model_file_exists(self):
		tpath = self.data_path + ".params" if not self.data_path.endswith(".params") else ""
		if not os.path.exists(tpath):
			#nope - not even the name of a parameters file
			raise Exception(f"Could not find data path or .params file:\n {self.data_path}")

	def to_json(self):
		res = {}
		res["env_name"] = self.env_name
		res["alpha"] =  str(self.alpha)
		res["gamma"] = str(self.gamma)
		res["epsilon"] = str(self.epsilon)
		res["batch_size"] = self.mini_batch_size
		res["target_refresh_rate_C"] = self.target_refresh_rate_C

		res["data_path"] = os.path.abspath(self.data_path)
		res["video_path"] = os.path.abspath(self.data_path)
		res["reward_shapers"] = [str(rs) for rs in self.reward_shapers]
		res["action_map"] = self.action_map

		return res
