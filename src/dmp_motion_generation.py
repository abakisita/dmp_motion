import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class dmp_motion_generation :

    def __init__(self, goal, y0, alpha, learned_parameters, time_step, alpha_x, tau, tau_learning):


        self.goal = goal
        self.y0 = y0
        self.alpha = alpha
        self.beta = alpha / 4
        self.learned_f = learned_parameters[0].T
        print(self.learned_f.shape, "hfsjsk")
        self.recorded_s = learned_parameters[1]
        self.time_step = tau * time_step / tau_learning
        self.alpha_x = alpha_x
        self.tau = tau

        self.pos = y0
        self.vel = np.zeros(y0.shape)
        self.vel = np.zeros(y0.shape)
        self.time_instances_passed = 0

    def canonical_system_output(self):

        time = self.time_instances_passed * self.time_step
        x = np.exp(-time * self.alpha_x)
        self.time_instances_passed += 1
        return x

    def interpolate_f(self, s):

        i = 0
        while not(s <= self.recorded_s[i] and s > self.recorded_s[i + 1]):
            i += 1
            if i >= len(self.recorded_s) - 1:
                return 0.0
                #return self.learned_f[:, self.learned_f.shape[1] - 1]
        f = (self.learned_f[:, i + 1] - self.learned_f[:, i]) * (s - self.recorded_s[i])/(self.recorded_s[i + 1] - self.recorded_s[i]) + self.learned_f[:, i]
        return f[np.newaxis].T

    def integrate_one_step(self, f):

        self.acc = (self.alpha * ( self.beta * (self.goal - self.pos) - self.vel * self.tau ) + (self.goal - self.y0) * f) / self.tau ** 2
        self.vel = self.vel + self.time_step * self.acc
        self.pos = self.pos + self.time_step * self.vel


    def integrate(self):

        pos_x = self.y0

        for i in range(200):
            s = self.canonical_system_output()
            f = self.interpolate_f(s)
            self.integrate_one_step(f * s)
            pos_x = np.hstack((pos_x,self.pos))

        return pos_x

    def plot(self):
        pos_x = self.integrate()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pos_x[0,:],pos_x[1,:],1)
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


