import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class dmp_moton_learning:

    def __init__(self, acc, vel, pos, K, D, time_step, canonical_constant, number_of_basis, basis_width):

        '''

        :param acc: n x m array containing acceleration (n is dimensions and m is number of data points)
        :param vel: n x m array containing velocity (n is dimensions and m is number of data points)
        :param pos: n x m array containing position (n is dimensions and m is number of data points)
        :param K: K gain (position)
        :param D: D gain (velocity)
        :param time_step: sampling time
        :param canonical_constant:
        :param number_of_basis: number of basis function
        :param basis_width: width of basis function
        '''

        self.x0 = pos[:,0]
        self.goal = pos[:,pos.shape[1] - 1]
        self.K = K
        self.D = D
        self.time_step = time_step
        self.canonical_constant = canonical_constant
        self.number_of_basis = number_of_basis
        self.basis_width = basis_width

        self.acc = acc
        self.vel = vel
        self.pos = pos


    def calculate_f(self):

        goal_matrix = np.ones((1,self.pos.shape[1])) * self.goal[0]
        for r in range(0, self.pos.shape[0] - 1):
            goal_matrix = np.vstack((goal_matrix,np.ones((1,self.pos.shape[1])) * self.goal[r + 1]))

        f = self.acc - self.K * (goal_matrix - self.pos) + self.D * self.vel

        diff = self.goal - self.x0

        for it in range(0, diff.shape[0]):

            f[it, :] *= diff[it]

        return f

    def canonical_system_output(self):

        time = np.array(self.time_step * np.linspace(0, self.pos.shape[1], self.pos.shape[1]))
        return np.exp(-time * self.canonical_constant)

    def calculate_centers(self):

        time = np.linspace(0, self.number_of_basis, self.number_of_basis) * \
               (self.pos.shape[1] * self.time_step * self.canonical_constant / self.number_of_basis)
        return np.exp(-time)

    def psi(self, h, s, c):

        diff = s - np.ones(s.shape[0]) * c
        return np.exp(-h * (s - c) ** 2)

    def learn_weights(self):

        centers = self.calculate_centers()
        s = self.canonical_system_output()
        psi_mat = []
        for c in centers:
            p = self.psi(self.basis_width, s, c)
            psi_mat.append(p / np.sum(p))

        psi_mat = np.array(psi_mat)
        psi_mat = psi_mat.T
        f = self.calculate_f()
        f = f.T
        weights = []
        for i in range(0, self.pos.shape[0]):
            w = np.linalg.pinv(psi_mat).dot(f[:,i][np.newaxis].T)
            weights.append(w[:,0])
        print(np.array(weights))

if __name__ == "__main__" :

    time = np.linspace(0, np.pi, 100)
    velx = np.sin(time)
    accx = np.cos(time)
    vely = time
    accy = np.ones(100) * (vely[1] - vely[0]) / 0.01
    posx = []
    posy = []
    x = 0.0
    y = 0.0
    for i in range(0, velx.shape[0]):
        x += velx[i] * 0.01
        y += vely[i] * 0.01
        posx.append(x)
        posy.append(y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(posx,posy,1)
    #plt.show()

    posx = np.array(posx)
    pos = np.vstack((posx, np.array(posy)))
    velx = np.array(velx)
    vel = np.vstack((velx, np.array(vely)))
    accx = np.array(accx)
    acc = np.vstack((accx, np.array(accy)))

    dmp_learn = dmp_moton_learning(acc, vel, pos, 2, 4, 0.01, 1, 10, 1)
    dmp_learn.learn_weights()
    dmp_learn.calculate_f()

    print(pos.shape)
