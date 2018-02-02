import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from promp.discrete_promp import DiscretePROMP
from promp.linear_sys_dyn import LinearSysDyn
from promp.promp_ctrl import PROMPCtrl


pickle_in = open(os.environ['ROOT_DIR'] + '/data/data.pkl',"rb")
data = pickle.load(pickle_in)


demos_list    = [data['steps']['states'][k][0][:,0] for k in range(100)]
Ddemos_list   = [data['steps']['states'][k][0][:,1] for k in range(100)]

def demo_generate_traj():

    #create a promb object by passing the data
    d_promp = DiscretePROMP(data=demos_list)
    d_promp.train()

    #add a via point
    # d_promp.add_viapoint(0.7, 5)
    # plt.scatter(0.7, 5, marker='*', s=100)

    #set the start and goal, the spatial scaling
    d_promp.set_start(demos_list[0][0])
    d_promp.set_goal(demos_list[0][-1])

    #add a via point
    # d_promp.add_viapoint(0.3, 2.25)
    # d_promp.add_viapoint(0.6, 2.25)
    # plt.scatter(0.7, 5, marker='*', s=100)

    for traj, traj_vel in zip(demos_list, Ddemos_list):
        plt.figure("ProMP-Pos")
        plt.plot(traj, 'k', alpha=0.2)
        plt.figure("ProMP-Vel")
        plt.plot(traj_vel, 'k', alpha=0.2)

    for _ in  range(1):

        pos_1, vel_1, acc_1 = d_promp.generate_trajectory(phase_speed=0.8,  randomness=1e-1)
        pos_2, vel_2, acc_2 = d_promp.generate_trajectory(phase_speed=1.,   randomness=1e-1)
        pos_3, vel_3, acc_3 = d_promp.generate_trajectory(phase_speed=1.33, randomness=1e-1)

        plt.figure("ProMP-Pos")
        plt.plot(pos_1, 'r')
        plt.plot(pos_2, 'g')
        plt.plot(pos_3, 'b')


        plt.figure("ProMP-Vel")
        plt.plot(vel_1, 'r')
        plt.plot(vel_2, 'g')
        plt.plot(vel_3, 'b')


def create_demo_traj():

    d_promp = DiscretePROMP(data=demos_list)
    d_promp.train()

    lsd = LinearSysDyn()

    state  = data['steps']['states'][0][0]
    action = data['steps']['actions'][0][0]

    promp_ctl = PROMPCtrl(promp_obj=d_promp, lin_dyn_obj=lsd)
    ctrl_cmds = promp_ctl.compute_ctrl_traj(state_list = state)

    plt.figure("Ctrl cmds")
    plt.plot(action, 'r')
    plt.plot(ctrl_cmds[:,0], 'g')
    


def main():

    demo_generate_traj()
    create_demo_traj()
    plt.show()


if __name__ == '__main__':
    main()