from akbinod.Utils.DecoratorBase import DecoratorBase, OutputTo
import time


class TimedFunction(DecoratorBase):
	def __init__(self, ignore, *, verbose=True, output_to=OutputTo.Console):
		super().__init__(ignore, verbose=verbose, output_to=output_to)

	def before(self, *args, **kwargs):
		"""Overide base class implementation."""
		super().before(*args, **kwargs)
		self.t1 = time.time()
		self.proc_time1 = time.process_time()

		return

	def after(self, returned_value):
		"""Overide base class implementation."""
		proc_time2 = time.process_time()
		t2 = time.gmtime(time.time() - self.t1)

		outp = f"""{self.func_name} processed in {time.strftime('%H:%M:%S', t2 )}"""
		if t2.tm_sec <= 1 or (proc_time2 - self.proc_time1) < 1:
			#if the number of seconds resolves to 1 or 0, twekas will
			#result in changes noticable only in process_time deltas
			#also, stop obsessing past the 5th decimal place
			outp += f"""\t[process_time: {round(proc_time2 - self.proc_time1,5)}]"""
		print(outp)
		super().after(returned_value)
		return
