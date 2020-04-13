import functools
from enum import Enum, auto

class OutputTo(Enum):
	Console = auto(),
	Property = auto()

def _wrap(obj):
	functools.wraps(obj.func)
	def call_proxy(*args, **kwargs):
		err = None
		value = None
		#this = kwargs["this"]
		try:
			#call the before function
			if not obj.before is None:
				obj.before(*args, **kwargs)
			#call the real function
			value = obj.func(*args, **kwargs)
		except Exception as error:
			err = error
		finally:
			#call the after function
			if not obj.after is None:
				obj.after(value)
			#if verbose and not err is None:
			#	logger.exception(err)
			if not err is None:
				raise err
		return value
	return call_proxy

class DecoratorBase():
	def __init__(self, ignore, *, verbose= True, output_to = OutputTo.Console):
		'''Called when first referenced by your code - not
		when the function is invoked.
		If init parameters were passed to your decorator,
		those are passed in here, but in this case the function
		to be decorated is not passed in. That will be passed
		in the invocation of __call__, and the inbovation expecys,
		__call__ to return a ref to the function that ultimately wraps the client function.
		The third and last call made by py is when the function
		is invoked from userland. This call goes to the fn that
		was returned by __call__
		I've forced a stupid param 'ignore' to
		force the workflow to be:
			1st call to __init__
			2nd call to __call__
			and the repeated final calls to the call proxy.

		FYI, this does work with vanilla functions - as well as with class methods.
		'''
		if callable(ignore):
			raise Exception("akbinod.Decorator must have a boolean 'ignore' as the first argument to the constructor.")

		self.output_to = output_to
		self.verbose = verbose
		#get things set up
		self.func = None

	def __call__( self, func):
		ret = None
		if self.func is None:
			#the init method was called without the function
			#this is the second setup call in the chain
			self.func = func

			#return the function that py should call when the
			#original function is actually called
			ret =  _wrap(self)
		else:
			#init was called with the function, this
			#call is to the function from userland code

			#now that we have forced the constructor to take
			#a positional argument, this section should never
			#be invoked, and we should never have the parameter shimmy
			#that this hare brained scheme induces
			print("awww, you shouldn't have... read the source")
		return ret

	@property
	def func(self):
		return self.__func

	@func.setter
	def func(self,func_to_call):
		self.__func = func_to_call
		if not func_to_call is None:
			self.__func_name = str(func_to_call)
			self.on_func_set()
		else:
			self.__func_name = ""
		return

	def inject_function_method(self, method_name,func, **keywords ):
		f = None
		try:
			f = self.func.__getattribute__(method_name)
			pass
		except:
			#ignore these
			pass
		if f is None:
			self.func.__setattr__(method_name, functools.partialmethod( func, **keywords))

		return

	def inject_object_method(self, objClient, method_name, func, **keywords):
		'''Inserts a method on the client object. Put keyword parameters you want to curry in **keywords '''
		f = None
		try:
			f = objClient.__getattribute__(method_name)
			pass
		except:
			#ignore these
			pass
		if f is None:
			objClient.__setattr__(method_name, functools.partial( func, **keywords))

		return

	def before(self, *args, **kwargs):
		'''Virtual. Called before the client function is called. '''
		if self.output_to == OutputTo.Console:
			print("********")
			print(f"""{self.func_name}""")
			#logging.basicConfig(level=logging.DEBUG)
			#logger = logging.getLogger(func_name)
		return

	def after(self, returned_value):
		'''Virtual. Called after the client function returns. '''
		print("********")
		return


	def on_func_set(self):
		'''Virtual. Callback for when the function is finalized.'''
		pass


	@property
	def func_name(self):
		return self.__func_name

