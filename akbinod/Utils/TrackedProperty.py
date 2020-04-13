from enum import Enum, auto
import functools
from akbinod.Utils.DecoratorBase import DecoratorBase, OutputTo
import copy

class TrackType(Enum):
	Last = auto(),
	List = auto()

class TrackedProperty(DecoratorBase):
	instances = []
	@staticmethod
	def register(t):
		TrackedProperty.instances.append(t)
		return
	@staticmethod
	def serialize_all_tracked(obj):
		'''Returns a json object with all tracked properties on obj'''
		obj_id = id(obj)
		ret = {}
		for i in range(len(TrackedProperty.instances)):
			inst = TrackedProperty.instances[i]
			inst.append_to(obj_id, ret)

		return ret

	def __init__(self, verbose,output_to=OutputTo.Console, track_type = TrackType.Last):
		super().__init__(verbose)
		self.track_type = track_type
		self.tracker = {}
		TrackedProperty.register(self)

	def on_func_set(self):
		self.inject_function_method("tp_serialize_to", self.serialize_property_to,this=self)
		return
	def before(self, *args, **kwargs):
		obj = args[0]
		val = args[1]
		key = id(obj)
		if not (key in self.tracker):
		#the first time we're seeing this concrete instance
			if self.track_type == TrackType.List:
				self.tracker[key] = []
			else:
				self.tracker[key] = None
			self.inject_object_method(obj,"serialize_all_tracked", TrackedProperty.serialize_all_tracked, obj=obj)

		if self.track_type == TrackType.Last:
			self.tracker[key] = val
		else:
			self.tracker[key].append(val)
		return


	def serialize_property_to(self, this, to_obj):
		'''Bound to a method on the tracked property, this returns the tracked values for just that one property on the concrete object.'''
		from_id = id(self)
		this.append_to(from_id, to_obj)
		return


	def append_to(self, from_id, to_obj):
		if from_id in self.tracker:
			#make a deep copy of whatever we are holding so things
			#can get mashed about later if the user wants.
			to_obj[self.func_name] = copy.deepcopy(self.tracker[from_id])
		return
