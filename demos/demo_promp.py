import numpy as np
import matplotlib.pyplot as plt
# from aml_lfd.promp.discrete_promp import DiscretePROMP

from aml_lfd.promp.promp_trial import DiscretePROMP
from aml_lfd.promp.promp_ctrl import PROMPCtrl


np.random.seed(0)

def demo_generate_traj():

    plt.figure("DiscretePROMP")

    # Generate and plot trajectory Data
    x = np.arange(0,1.01,0.01)           # time points for trajectories
    nrTraj=30                            # number of trajectoreis for training
    sigmaNoise=0.02                       # noise on training trajectories
    A = np.array([.2, .2, .01, -.05])
    X = np.vstack( (np.sin(5*x), x**2, x, np.ones((1,len(x))) ))
    Y = np.zeros( (nrTraj,len(x)) )

    demos_list = []
    for traj in range(0, nrTraj):
        sample = np.dot(A + sigmaNoise * np.random.randn(1,4), X)[0]
        plt.plot(sample, 'k')
        demos_list.append(sample)


    #create a promb object by passing the data
    d_promp = DiscretePROMP(data=demos_list)
    d_promp.train()

    #add a via point
    # d_promp.add_viapoint(0.7, 5)
    # plt.scatter(0.7, 5, marker='*', s=100)

    #set the start and goal, the spatial scaling
    d_promp.set_start(0.2)
    d_promp.set_goal(-0.1)

    for _ in  range(100):

        plt.plot(d_promp.generate_trajectory(phase_speed=0.5, randomness=1e-3), 'r')
        plt.plot(d_promp.generate_trajectory(phase_speed=1., randomness=1e-3), 'g')
        plt.plot(d_promp.generate_trajectory(phase_speed=2., randomness=1e-3), 'b')

    plt.legend()
    plt.show()



def demo_ctrl_traj():

    A = np.array([.2, .2, .01, -.05])
    B = np.random.randn(4,2)

    x0 = np.zeros(4)

    demos_list = []
    T = 100
    n_demos = 30
    sigma_noise = 0.01

    u = np.random.randn(2, T)

    for _ in range(n_demos):
        traj = np.zeros([4,T])

        for k in range(T-1):

            traj[:, k+1] = np.dot( (A + sigma_noise * np.random.randn(4)), traj[:, k]) + np.dot(B, u[:, k])

        demos_list.append(traj[0, :].copy())


    #create a promb object by passing the data
    d_promp = DiscretePROMP(data=demos_list)
    d_promp.train()

    c_promp = PROMPCtrl(promp_obj=d_promp, A=A, B=B)

    u_computed = np.zeros(T)

    for t in range(T):
        u_computed[t] = c_promp.compute_gains(t, add_noise=False)


    print "Initial u \n", u
    print "Computed u \n", u_computed



def main():

    demo_ctrl_traj()
    # demo_generate_traj()


if __name__ == '__main__':
    main()


