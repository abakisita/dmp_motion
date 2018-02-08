import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class dmp_motion_generation :

    def __init__(self, goal, x0, K, D, time_step):

        self.goal = goal[np.newaxis].T
        self.x0 = x0[np.newaxis].T
        self.K = K
        self.D = D
        self.time_step = time_step

        self.acc = np.zeros(goal.shape)
        self.vel = np.zeros(goal.shape)
        self.pos = x0

        print(self.acc.shape)
        print(self.pos)
        print(self.time_step)

    def calculate_f(self):
        raise NotImplementedError()

    def integrate_one_step(self):

        self.acc = self.K * (self.goal - self.pos) - self.D * self.vel
        self.vel = self.vel + self.time_step * self.acc
        self.pos = self.pos + self.time_step * self.vel

    def integrate(self):
        pos_seq = self.pos
        for i in range(1, 1000):
            self.integrate_one_step()
            pos_seq = np.hstack((pos_seq, self.pos[0]))
        return pos_seq

    def plot_dmp(self):

        pos = self.integrate()
        print(pos.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pos[0,:],pos[1,:],pos[2,:])
        plt.show()


if __name__ == "__main__":
    goal = np.array([[1.2],
                     [2.3],
                     [3.5]])
    x0 = np.array([[0],
                   [0],
                   [0]])
    dmp = dmp_motion_generation(goal, x0, 2, 4, 0.01)
    dmp.plot_dmp()




