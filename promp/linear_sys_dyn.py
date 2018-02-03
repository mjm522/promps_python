import numpy as np


class LinearSysDyn():
    """
    A simple linear system dynamics
    a system of mass 1kg and 1D state.
    This class is a general linear system 
    if a different system matrix (A) and different control matrix (B) is passed
    """

    def __init__(self, A=None, B=None, c=None, dt=0.005, mass=1.):
        """
        Constructor of the class
        Args:
        A = System matrix = > shape: [state_dim x state_dim]
        B = Control matrix = > shape: [state_dim x action_dim]
        c = Sysem drift matrix => shape: [state_dim x action_dim]
        dt = time step
        mass = mass of the body
        """

        #system matrix
        if A is None:
            self._A = np.array([ [0.,1.], [0., 0.] ])
        else:
            self._A = A

        #control matrix
        if B is None:
            self._B = np.array([ [0.], [1.] ])
        else:
            self._B = B

        #drift vector
        if c is None:
            self._c = np.array([ [0.], [0.] ])
        else:
            self._c = c
        
        self._dt = dt

        self._mass = mass

        self._state_dim, self._action_dim = self._B.shape #position + velocity


    def compute_next_states(self, state_list, action_list):
        """
        A simple one step integration 
        for computing next state, the operations are done for
        the entire statelist and action list
        Args: 
        state_list = array of states shape => [time_steps, state_dim]
        action_list = array of states shape => [time_steps, action_dim]
        """

        #assuming action are forces
        acc = (action_list*1./self._mass).squeeze()

        d_state_list = np.zeros_like(state_list)

        #assign the velocity
        d_state_list[:, 0:self._state_dim/2] = state_list[:, 0:self._state_dim/2]
        #assign the acceleration
        d_state_list[:, self._state_dim/2:] = acc

        #euler integration position
        state_new_list = state_list + self._dt * d_state_list

        #euler integration velocity
        state_new_list[:, self._state_dim/2:] += 0.5*acc*self._dt**2

        return state_new_list

