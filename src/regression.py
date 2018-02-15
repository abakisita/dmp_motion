import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class gaussian_regression():

    def __init__(self, f):

        #self.s = np.array(s)[np.newaxis].T
        #self.psi = np.diag(psi)
        self.f = np.array(f)[np.newaxis].T
        self.runtime = 1000
        self.time_step = 0.01
        self.canonical_constant = -np.log(0.01)
        self.number_of_basis = 30
        self.time = np.linspace(0, self.runtime * self.time_step * self.canonical_constant, self.runtime)
        self.s__ = np.exp(-self.time)
        self.function = np.sin(self.s__ * 10)
        self.f = self.function

    def learn_w(self, psi):
        s = np.array(self.s)[np.newaxis].T
        psi = np.diag(psi)
        w = s.T.dot(psi.dot(self.f)) / s.T.dot(psi.dot(s))
        return w

    def canonical_system_output(self):


        x = np.exp(-self.time)
        return x

    def psi(self, s, c):

        h = 110
        return np.exp(-h * (s - c) ** 2)

    def learn_weights(self):

        time = np.linspace(0, self.runtime * self.time_step * self.canonical_constant, self.number_of_basis)
        #self.centers = np.exp(-time)
        self.centers = np.linspace(0, 1, self.number_of_basis)
        self.s = self.canonical_system_output()
        weights = []
        for c in self.centers:
            plt.plot(self.s, self.psi(self.s, c))

        for c in self.centers:
            w = self.learn_w(self.psi(self.s, c))
            weights.append(w[0,0])

        print(weights)
        plt.plot(self.s, self.function)
        return np.array(weights)[np.newaxis].T


    def recon(self, we):

        fn = []
        for s in self.s:
            psi = self.psi(s, self.centers)
            f = psi.dot(we) / np.sum(psi)
            fn.append(f)

        plt.plot(self.s, fn)




r = gaussian_regression(0.0)
w = r.learn_weights()
print(w)
r.recon(w)
plt.show()