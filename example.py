from environments.base import MujocoEnv
from tsc.tsc import TransitionStateClustering
import numpy as np

m = MujocoEnv(visualize=True)

s0 = (m.model.data.qpos,
	 m.model.data.qvel,
	 m.model.data.qacc)

m.initialize(s0)

for i in range(0,100):
	m.applyControl(lambda s,t: None) #apply no control

a = TransitionStateClustering(window_size=3, normalize=True, pruning=0.0,delta=-1)
a.addDemonstration(np.squeeze([satups[0][0] for satups in m.trajectory]))
a.addDemonstration(np.squeeze([satups[0][0]+0.01*np.random.randn(15,1) for satups in m.trajectory]))

a.fit()
