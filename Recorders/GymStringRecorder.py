from Recorders.BaseRecorder import BaseRecorder
from string_recorder import StringRecorder
# from matplotlib import font_manager as fm
from matplotlib.font_manager import findfont, FontProperties
import os

class GymStringRecorder(BaseRecorder):
	def __init__(self, path, env):
		super().__init__(path, "gif", env)

		# fmg = fm.FontManager()
		# normal = fmg.findfont("Courier New 5.00.2x")
		font = findfont(FontProperties(family=['monospace'],size='large'))
		self.recorder = StringRecorder(font=font)

	def capture_frame(self,print = False):
		frame = self.env.render(mode='ansi')
		if print: print(frame)
		self.recorder.record_frame(frame,)

		return

	def end(self, score, steps):
		#close out the recording
		try:
			self.recorder.make_gif(self.new_path(score,steps),speed=.8)
		except :
			#ignore these  writing errors
			pass

		return