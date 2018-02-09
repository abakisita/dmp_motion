import numpy as np
import matplotlib.pyplot as plt

class dmp_moton_learning:

    def __init__(self, acc, vel, pos, K, D, time_step, canonical_constant, number_of_basis, basis_width):

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

        print(self.acc.shape)
        print(self.pos)
        print(self.time_step)

    def calculate_f(self):

        goal_matrix = np.ones((1,self.pos.shape[1])) * self.goal[0,0]
        for r in range(0, self.pos.shape[0] - 1):
            goal_matrix = np.hstack((goal_matrix,np.ones((1,self.pos.shape[1])) * self.goal[r + 1,0]))

        f = self.acc - self.K * (goal_matrix - self.pos) + self.D * self.vel

        diff = self.goal - self.x0

        for it in range(0, diff.shape[0]):

            f[it, :] *= diff[it]

        print(goal_matrix)

    def canonical_system_output(self):

        time = np.array(self.time_step * range(0, self.pos.shape[1]))
        return np.exp(-time * self.canonical_constant)

    def calculate_centers(self):

        time = np.linspace(0, self.number_of_basis, self.number_of_basis) * \
               (self.pos[1] * self.time_step * self.canonical_constant / self.number_of_basis)
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
            psi_mat.append(p)

        print(psi_mat)


