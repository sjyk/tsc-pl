import mujoco_py
from mujoco_py import mjviewer, mjcore
from mujoco_py import glfw
import six

"""
This module implements the base environments and
all other environments extend this.
"""


"""
The base environment
"""
class BaseEnv(object):

	def __init__(self, visualize=False):
		self.visualize = visualize
		super(BaseEnv, self).__init__()

	"""
	Initializes the environment at state s0
	"""
	def initialize(self, s0):
		self.trajectory = [ (s0, None) ]
		self.setStateTo(s0)

		if self.visualize:
			self.updatePlot()


	"""
	Applies the control policy to the environment, policy
	is a lambda: S x T -> A
	"""
	def applyControl(self, policy):
		observed_state = self.getCurrentObservedState()
		observed_time = self.getCurrentTime()
		action = policy(observed_state, observed_time)
		new_state = self.dynamicStep(action)
		self.trajectory.append( (observed_state, action) )

		if self.visualize:
			self.updatePlot()

	"""
	Returns the current time-step of execution
	"""
	def getCurrentTime(self):
		return len(self.trajectory)

	"""
	Sets the state to s
	"""
	def setStateTo(self, s):
		raise NotImplemented("Implement the function setStateTo(self, newstate)")

	"""
	Gets the current state
	"""
	def getCurrentState(self):
		raise NotImplemented("Implement the function getCurrentState(self)")

	"""
	Gets the current state observed by the agent
	"""
	def getCurrentObservedState(self):
		raise NotImplemented("Implement the function getCurrentObservedState(self)")

	"""
	Steps the state given a control, returns a new state
	"""
	def dynamicStep(self, a):
		raise NotImplemented("Implement the function step(self, newstate)")

	"""
	Updates any visualization
	"""
	def updatePlot(self):
		raise NotImplemented("Implement the function updatePlot(self)")



"""
The Mujoco Environment extends the baseenv
"""
class MujocoEnv(BaseEnv):

	#state is a three tuple of (pos, vel, acc)

	def __init__(self, path='models/ant.xml', visualize=False):
		self.model = mujoco_py.MjModel(path)

		if visualize:
			self.viewer = mjviewer.MjViewer(visible=True,
                            	   		init_width=1000,
                            	   		init_height=1000)
			self.viewer.start()
			self.viewer.set_model(self.model)
		
		super(MujocoEnv, self).__init__(visualize)


	#override set state 
	def setStateTo(self, s):
		self.model.data.qpos = s[0]
		self.model.data.qvel = s[1]
		self.model.data.qacc = s[2]

	#override get state 
	def getCurrentState(self):
		return (self.model.data.qpos,
				self.model.data.qvel,
				self.model.data.qacc)

	#override get state 
	def getCurrentObservedState(self):
		return self.getCurrentState()

	#override dynamic step (allows for autonomous)
	def dynamicStep(self, a):
		
		if a != None:
			self.model.data.ctrl = a

		self.model.step()
		self.model.forward()
		return self.getCurrentState()

	#override update plot
	def updatePlot(self):
		if self.viewer != None:
			self.viewer.loop_once()


