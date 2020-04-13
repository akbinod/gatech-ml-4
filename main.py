import gym
import gym.utils

import akbinod.Utils.Plotting as plt
from akbinod import DecayedParameter, UndecayedParameter, TimedFunction
from FrozenLake import FrozenLake
from akbinod.Utils.LittmanAlpha import LittmanAlpha
from akbinod.Utils import Plotting
from Learners.LearnerParams import LearnerParams
from Agent import Agent
from Learners.FrozenLakeLearner import FrozenLakeLearner
from Recorders import GymStringRecorder
from Learners.defs import _MAPS, _ACTIONS
from Learners.FLRewardShaper import FLRewardShaper
from akbinod.Utils.TimedFunction import TimedFunction
from Learners.NChainLearner import NChainLearner
from Learners.NCRewardShaper import NCRewardShaper

action_map = {
	"0":"left"
	,"1":"down"
	,"2":"right"
	,"3":"up"
}

nc_action_map = {
	"0":"forward"
	,"1":"backward"
}

def frozenLake(size):
	path = f"./Data/fl_{str(size)}_vi"

	params = LearnerParams("FrozenLake", epsilon=None,gamma=0.95
						,data_path=path ,video_root="./Data")
	params.max_iterations = 100_000
	params.map_key = f"{str(size)}x{str(size)}"
	params.theta = 0.05
	params.seed = 903549491

	f = FrozenLake(params)
	f.vi()
	pvi = Plotting.Plot("Value Iteration Convergence","episodes","error", f.convergence_tracker)

	params.data_path = f"./Data/fl_{str(size)}_pi"
	f = FrozenLake(params)
	f.pi()
	ppi = Plotting.Plot("Policy Iteration Convergence","episodes","policy error", f.convergence_tracker)

	pvi.show()
	ppi.show()

TimedFunction(True)
def frozen_lake_q(train, size, run_name = 'frozen_lake'):

	map_key = f"{str(size)}x{str(size)}"
	map = _MAPS[map_key]
	with gym.make('FrozenLake-v0', is_slippery=True, desc=map).unwrapped as env:
		env.seed(903549491)
		min_run = 10000
		max_run =  min_run * 500

		print(f"Running grid {size} over {max_run} iterations.")

		param_file = ""
		data_path='./Data/' if train else './Data/' + param_file
		video_root='./Data/'
		# epsilon = UndecayedParameter(0.20)
		epsilon = DecayedParameter(0.99995, 0.0005, max_run)
		# alpha = LittmanAlpha(anticipatedStepsT=max_run,decay=0.9)
		alpha = DecayedParameter(0.99995, 0.0005, max_run)
		gamma = 0.9995


		# need a reward shaper to break ties in exploitation
		# and learning to avoid holes
		reward_shaper = FLRewardShaper(map)
		# go with canned defaults for other hyper params
		# alpha will be picked
		learner_params = LearnerParams(run_name
										, data_path=data_path, video_root=video_root
										, epsilon 	= epsilon
										, alpha 	= alpha
										, gamma		= gamma
										, reward_shaper = reward_shaper)
		learner_params.action_map = action_map
		# used in the convergence check
		learner_params.convergence_theta = 0.001
		rec = GymStringRecorder(learner_params.video_path,env)

		learner = FrozenLakeLearner(env, learner_params)
		agent = Agent(learner, run_name=run_name,target_moving_avg=75, recorder=rec)

		if train:
			agent.train(min_run,max_run)
			agent.convergence_plot.show()

			# dump the convergence tracker
			with open("./data/" + run_name + ".csv","w+") as f:
				for i in learner.convergence_tracker:
					f.write(str(round(abs(i),5))+"\n")

			agent.epsilon_plot.show()
			agent.train_plot.show()
			agent.train_steps_plot.show()
			# if not agent.loss_plot is None:
			# 	agent.loss_plot.show()
			p = plt.Plot("Alpha Decay", "episodes", "alpha", learner.params.alpha.usage)
			p.show()
			p = agent.state_visits_plot
			if not p is None:
				p.show()
		else:
			agent.play(min_run, max_run)
			agent.play_plot.show()


	return

TimedFunction(True)
def nchain_q(train, run_name = 'chain'):

	with gym.make('NChain-v0').unwrapped as env:
		env.seed(903549491)
		min_run = 1000
		max_run =  min_run * 10000

		print(f"Running NChain over {max_run} iterations.")

		param_file = ""
		data_path='./Data/' if train else './Data/' + param_file
		video_root='./Data/'
		# epsilon = UndecayedParameter(0.20)
		epsilon = DecayedParameter(0.99995, 0.0005, max_run)
		# alpha = LittmanAlpha(anticipatedStepsT=max_run,decay=0.9)
		alpha = DecayedParameter(0.99995, 0.0005, max_run)
		gamma = 0.9995


		# # need a reward shaper to break ties in exploitation
		# # and learning to avoid holes
		reward_shaper = None
		# reward_shaper = NCRewardShaper(5)
		# go with canned defaults for other hyper params
		learner_params = LearnerParams(run_name
										, data_path=data_path, video_root=video_root
										, epsilon 	= epsilon
										, alpha 	= alpha
										, gamma		= gamma
										, reward_shaper = reward_shaper)
		learner_params.action_map = nc_action_map
		# used in the convergence check
		learner_params.convergence_theta = 0.001
		rec = None

		learner = NChainLearner(env, learner_params)
		agent = Agent(learner, run_name=run_name,target_moving_avg=4, recorder=rec)

		if train:
			agent.train(min_run,max_run)
			agent.convergence_plot.show()

			# dump the convergence tracker
			with open("./data/" + run_name + ".csv","w+") as f:
				for i in learner.convergence_tracker:
					f.write(str(round(abs(i),5))+"\n")

			agent.epsilon_plot.show()
			agent.train_plot.show()
			agent.train_steps_plot.show()
			# if not agent.loss_plot is None:
			# 	agent.loss_plot.show()
			p = plt.Plot("Alpha Decay", "episodes", "alpha", learner.params.alpha.usage)
			p.show()
			p = agent.state_visits_plot
			if not p is None:
				p.show()
		else:
			agent.play(min_run, max_run)
			agent.play_plot.show()


	return

def main():

	# frozenLake(32)

	# size = 16
	# frozen_lake_q(True,size)

	nchain_q(True)
if __name__ == "__main__":
	err = None
	try:
		main()
	except Exception as e:
		err = e
	finally:
		print("*****************")
		if not err is None:
			raise err