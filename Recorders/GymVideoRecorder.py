from Recorders.BaseRecorder import BaseRecorder
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import os

class GymVideoRecorder(BaseRecorder):
	def __init__(self, path, env):
		super().__init__(path,"mp4",env)
		self.recorder = VideoRecorder(self.env, self.path)

