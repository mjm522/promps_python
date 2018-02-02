import numpy as np


class PROMPCtrl(object):
    """
    This class computes the feedback and feedforward gains of the a PROMP in
    closed loop. For more reference, see: https://link.springer.com/article/10.1007/s10514-017-9648-7
    """

    def __init__(self, promp_obj, lin_dyn_obj):

        """
        Constructor of the class:
        Args:
        A = linearised system dynamics matrix
        B = linearised system control matrix
        D_mu = 
        D_cov= 
        promb_obj = Instatiation of the discrete promp class
        """
        #dynamics object
        self._dynamics = lin_dyn_obj

        #promp object
        self._promp = promp_obj

        #system matrix
        self._A = self._dynamics._A

        #control matrix
        self._B = self._dynamics._B

        #time step
        self._dt = self._dynamics._dt

        self._Phi   = self._promp._Phi
        self._PhiD  = self._promp._PhiD
        self._PhiDD = self._promp._PhiDD

        self._sigma_W = self._promp._sigma_W
        self._mean_W  = self._promp._mean_W

       
    def update_system_matrices(self, A, B):
        """
        Update the system matrices 
        this is for the purpose of adding time varying 
        system matrices
        """
        self._A = A
        self._B = B


    def get_basis(self, t):
        """
        This function creates a basis and Dbasis
        basis  = [Phi; PhiD]
        Dbasis = [PhiD; PhiDD]
        """
        return np.vstack([self._Phi[:, t], self._PhiD[:, t]]), np.vstack([self._PhiD[:, t], self._PhiDD[:, t]])


    def compute_gains(self, t, add_noise=True):
        """
        the control command is assumed to be of type
        u = Kx + k + eps
        """
        #get the basis funtion at a time step
        basis, Dbasis = self.get_basis(t)

        #equation 12
        Sigma_t =  np.dot(np.dot(basis, self._sigma_W), basis.T)

        #part 1 equation 46
        B_pseudo = np.linalg.pinv(self._B)

        #part 2 equation 46
        tmp1 = np.dot(np.dot(Dbasis, self._sigma_W), basis.T)

        #part 3 equation 46
        tmp2 = np.dot(self._A, Sigma_t)

        #compute feedback gain; complete equation 46
        K = np.dot( np.dot(B_pseudo, (tmp1-tmp2)), np.linalg.inv(Sigma_t))

        #part 1 equation 48
        tmp3 = np.dot(Dbasis, self._mean_W)

        #part 2 equation 48
        tmp4 = np.dot( (self._A + np.dot(self._B, K)), np.dot(basis, self._mean_W) )

        #compute feedforward gain; complete equation 48
        k = np.dot(B_pseudo, (tmp3-tmp4))

        return K, k


    def compute_gain_traj(self):
        """
        This function is to compute the entire gain trajectory
        of a given state distribution
        """
        time_steps = self._Phi.shape[1]
        state_dim, action_dim = self._B.shape

        K_traj = np.zeros([time_steps, state_dim, state_dim])
        k_traj = np.zeros([time_steps, action_dim])

        for t in range(time_steps):

            K_traj[t, :, :], k_traj[t, :] = self.compute_gains(t)

        return K_traj, k_traj

    def compute_control_cmd(self, t, state):
        """
        This function is compute the specific control
        command at a time step t 
        Args: 
        t : time step
        state : state for which control command needs to be computed
        """

        K, k = self.compute_gains(t)

        return np.dot(K, state) + k


    def compute_ctrl_traj(self, state_list):
        """
        This function computes an entire
        control sequence for a given state list
        Args:
        state_list for which control has to be computed
        this assumes that len(state_list) = timesteps in the basis function
        """

        time_steps = self._Phi.shape[1]
        _, action_dim = self._B.shape

        ctrl_cmds =  np.zeros([time_steps, action_dim])

        for t in range(time_steps):

            ctrl_cmds[t, :] = self.compute_control_cmd(t, state_list[t,:])

        return ctrl_cmds



