import numpy as np


class LinearSysDyn():

    def __init__(self):

        #system matrix
        self._A = np.array([ [0.,1.], [0., 0.] ])

        #control matrix
        self._B = np.array([ [0.], [1.] ])

        #drift vector
        self._c = np.array([ [0.], [0.] ])
        
        action_noise = 12.25
        self._H = np.array([ [0., 0.], [0., action_noise] ])
        self._noise_std = 3.5
        self._dt = 0.005
        self._time_steps = 200
        self._mass = 1.
        self._state_dim = 2 #position + velocity
        self._action_dim = 1

    def compute_next_state(self, state, action):
        #assuming action are forces

        acc = (action*1./self._mass).squeeze()

        d_state = np.zeros_like(state)

        d_state[:, 0] = state[:, 0]
        d_state[:, 1] = acc

        state_new = state + self._dt * d_state

        state_new[:, 1] += 0.5*acc*self._dt**2

        return state_new
