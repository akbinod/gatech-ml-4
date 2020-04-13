import os

class BaseRecorder():
	def __init__(self, path, ext, env):
		self.env = env
		self.__path = path
		self.__episode = 0
		# must have this extension or the video recorder barfs
		self.__extension = ext

	def begin(self, ep_num):
		self.__episode = ep_num
		path = self.path
		if os.path.exists(path):
			#delete the old recording for this episode if it exists
			os.remove(path)
		# capture the first frame (starting spot right after a reset)
		self.capture_frame()

	@property
	def path(self):
		# build out the path with all the pieces we have
		path = self.__path + f"""_episode_{self.episode}.{self.extension}"""
		return path

	@property
	def extension(self):
		return self.__extension

	@property
	def episode(self):
		return self.__episode

	def capture_frame(self):
		self.recorder.capture_frame()
		return

	def new_path(self, score, steps):
		new_path = self.path.replace(self.extension,f"""_score_{int(score)}_steps_{steps}.{self.extension}""")
		if os.path.exists(self.path):
			os.rename(self.path, new_path)
		return new_path


	def end(self, score, steps):
		# close out the recording
		try:
			self.recorder.close()
			self.new_path(score, steps)
		except :
			#ignore these  writing errors
			pass
		return
