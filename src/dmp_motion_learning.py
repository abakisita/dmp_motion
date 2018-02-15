import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dmp_motion_generation

class dmp_moton_learning:

    def __init__(self, acc, vel, pos, alpha, time_step, alpha_x, tau):

        '''

        :param acc: n x m array containing acceleration (n is dimensions and m is number of data points)
        :param vel: n x m array containing velocity (n is dimensions and m is number of data points)
        :param pos: n x m array containing position (n is dimensions and m is number of data points)
        :param alpha: alpha
        :param beta: beta
        :param time_step: sampling time
        :param canonical_constant: canonical constant
        '''

        self.y0 = pos[:,0]
        self.goal = pos[:,pos.shape[1] - 1]
        self.alpha = alpha
        self.beta = alpha / 4
        self.time_step = time_step
        self.alpha_x = alpha_x
        self.tau = tau

        self.acc = acc
        self.vel = vel
        self.pos = pos

        self.time_instances_passed = 0

    def get_f(self):
        self.f = []
        self.s = []
        goal = self.pos[:, -1]

        for i in range(0, self.pos.shape[1]):
            f = self.acc[:,i] * self.tau ** 2 - self.alpha * ( self.beta * (goal - self.pos[:,i]) - self.vel[:,i] * self.tau)
            s = self.canonical_system_output()
            self.f.append(f / (s * (goal - self.y0)))
            self.s.append(s)
            self.time_instances_passed += 1
        self.f = np.array(self.f)
        self.time_instances_passed = 0
        return self.f, np.array(self.s)

    def canonical_system_output(self):

        time = self.time_instances_passed * self.time_step
        x = np.exp(-time * self.alpha_x)
        return x


if __name__ == "__main__" :

    time = np.linspace(0, np.pi, 100)
    velx = np.sin(time)
    accx = np.cos(time)
    vely = np.hstack((np.linspace(0, 50, 50), np.linspace(51, 100, 50))) * 0.02

    accy = [0]
    for i in range(0, vely.shape[0] - 1):
        accy.append(vely[i + 1] - vely[i])
    print(len(accy))

    posx = []
    posy = []
    x = 0.0
    y = 0.0
    for i in range(0, velx.shape[0]):
        x += velx[i] * np.pi / 100
        y += vely[i] * np.pi / 100
        posx.append(x)
        posy.append(y)



    posx = np.array(posx)
    pos = np.vstack((posx, np.array(posy)))
    velx = np.array(velx)
    vel = np.vstack((velx, vely))
    accx = np.array(accx)
    acc = np.vstack((accx, np.array(accy)))
    print(pos.shape)
    print(vel.shape)
    print(acc.shape)
    dmp_learn = dmp_moton_learning(acc, vel, pos, 4.0, np.pi/100, -np.log(0.01), 1.5)
    f, s = dmp_learn.get_f()

    #goal = pos[:,pos.shape[1] - 1][np.newaxis].T
    goal = np.array([[8.0],
                     [8.0]])

    x0 = np.array([[1],
                   [1]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[0,:],pos[1,:],1)
    dmp_gen = dmp_motion_generation.dmp_motion_generation(goal, x0, 4.0, (f, s), np.pi/100, -np.log(0.01), 1.5, 1.5)
    dmp_gen.plot()

