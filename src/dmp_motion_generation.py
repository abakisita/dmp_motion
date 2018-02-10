import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class dmp_motion_generation :

    def __init__(self, goal, x0, K, D, time_step, centers, weights, canonical_constant, number_of_basis, basis_width, f):

        self.goal = goal[np.newaxis].T
        self.x0 = x0[np.newaxis].T
        self.K = K
        self.D = D
        self.time_step = time_step
        self.centers = centers
        self.weights = weights
        self.number_of_basis = number_of_basis
        self.basis_width = basis_width
        self.canonical_constant = canonical_constant
        self.f = f
        self.__f = []

        self.acc = np.zeros(goal.shape)
        self.vel = np.zeros(goal.shape)
        self.pos = x0
        self.time_instances_passed = 0


    def canonical_system_output(self):

        time = self.time_instances_passed * self.time_step
        return np.exp(-time * self.canonical_constant)

    def psi(self, h, s, c):

        return np.exp(-h * (s - c) ** 2)

    def calculate_f(self):

        s = self.canonical_system_output()
        psi = self.psi(self.basis_width, s, self.centers)
        psi = psi[np.newaxis].T
        f = self.weights.dot(psi) * s /np.sum(psi)
        f = f.T
        diff = self.goal - self.x0
        diff = diff[0,:,0]
        return f.dot(np.diag(diff))

    def integrate_one_step(self):
        diff = self.goal - self.x0
        diff = diff[0,:,0]
        #_f = self.f[:,self.time_instances_passed].T.dot(np.diag(diff))
        _f = self.calculate_f()
        self.acc = self.K * (self.goal - self.pos) - self.D * self.vel + _f[np.newaxis].T
        print(self.acc)
        self.vel = self.vel + self.time_step * self.acc
        self.pos = self.pos + self.time_step * self.vel
        self.time_instances_passed += 1

    def integrate(self):
        pos_seq = self.pos
        for i in range(0, 500):
            self.integrate_one_step()
            pos_seq = np.hstack((pos_seq, self.pos[0]))
        return pos_seq

    def plot_dmp(self):

        pos = self.integrate()
        print(pos.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pos[0,:],pos[1,:],1)
        plt.show()

'''

if __name__ == "__main__":
    goal = np.array([[1.2],
                     [2.3],
                     [3.5]])
    x0 = np.array([[0],
                   [0],
                   [0]])
    dmp = dmp_motion_generation(goal, x0, 2, 4, 0.01)
    #dmp.plot_dmp()
'''


