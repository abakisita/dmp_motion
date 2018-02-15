import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




class dmp_moton_learning:

    def __init__(self, acc, vel, pos, alpha, beta, time_step, alpha_x, tau):

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

        self.x0 = 0.0
        self.goal = 2
        self.alpha = alpha
        self.beta = alpha / 4
        self.time_step = time_step
        self.alpha_x = alpha_x
        self.tau = tau

        self.iteration = 100
        self.acc = 0.0
        self.vel = 0.0
        self.pos = 0.0
        self.posl = pos
        self.vell = vel
        self.accl = acc
        self.time_instances_passed = 0


    def canonical_system_output(self):

        time = self.time_instances_passed * self.time_step
        x = np.exp(-time * self.alpha_x)
        return x


    def get_f(self):
        self.f = []
        self.s = []
        goal = self.posl[self.posl.shape[0] - 1]
        for i in range(0, self.posl.shape[0]):
            f = self.accl[i] * self.tau ** 2 - self.alpha * ( self.beta * (goal - self.posl[i]) - self.vell[i] * self.tau)
            s = self.canonical_system_output()
            self.f.append(f / (s * (goal - self.x0)))
            self.s.append(s)
            self.time_instances_passed += 1

        self.time_instances_passed = 0

    def interpolate_f(self, s):

        i = 0
        while not(s <= self.s[i] and s > self.s[i + 1]):
            i += 1
            if i >= len(self.s) - 1:
                return self.f[len(self.f) - 1]
        f = (self.f[i + 1] - self.f[i]) * (s - self.s[i])/(self.s[i + 1] - self.s[i]) + self.f[i]
        return f

    def integrate_one_step(self, f):
        self.acc = (self.alpha * ( self.beta * (self.goal - self.pos) - self.vel * self.tau ) + (self.goal - self.x0)*f) / self.tau ** 2
        self.vel = self.vel + self.time_step * self.acc
        self.pos = self.pos + self.time_step * self.vel
        self.time_instances_passed += 1


    def integrate(self , f):
            self.time_instances_passed = 0
            pos_seq = [self.pos]
            if len(f) == 1:
                for i in range(0, 500):
                    self.integrate_one_step(0.0)
                    pos_seq.append(self.pos)
                return pos_seq
            else:
                self.acc = 0.0
                self.pos = 0.0
                self.vel = 0.0
                self.goal = 2

                for i in range(0, len(f)):
                    s = self.canonical_system_output()
                    f = self.interpolate_f(s)
                    self.integrate_one_step(f * s)
                    pos_seq.append(self.pos)
                for i in range(0, 300):
                    self.integrate_one_step(0.0)
                    pos_seq.append(self.pos)
                return pos_seq


    def plot(self):
        po = self.integrate([0.0])
        po_f = self.integrate(self.f)

        plt.plot(np.linspace(0, len(po_f) * self.time_step , len(po_f)), po_f)
        plt.plot(np.linspace(0, len(po) * self.time_step , len(po)), po)
        plt.show()


time = np.linspace(0, np.pi, 100)
velx = np.sin(time)
accx = np.cos(time)
vely = np.sin(time)
accy = np.cos(time)
posx = []
posy = []
x = 0.0
y = 0.0
for i in range(0, velx.shape[0]):
    x += velx[i] * np.pi / 100
    y += vely[i] * 0.001
    posx.append(x)
    posy.append(y)

posx = np.array(posx)
velx = np.array(velx)
accx = np.array(accx)

plt.plot(time , posx)
#plt.show()

dmp = dmp_moton_learning(accx, velx, posx, 4.0, 1.0, np.pi / 100, -np.log(0.01), 1.5)
dmp.get_f()
dmp.plot()
