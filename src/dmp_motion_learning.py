import numpy as np
import matplotlib.pyplot as plt

class dmp_moton_learning:

    def __init__(self, acc, vel, pos, K, D, time_step, canonical_constant):

        self.goal = pos[:,0]
        self.x0 = pos[:,pos.shape[1] - 1]
        self.K = K
        self.D = D
        self.time_step = time_step
        self.canonical_constant = canonical_constant

        self.acc = acc
        self.vel = vel
        self.pos = pos

        print(self.acc.shape)
        print(self.pos)
        print(self.time_step)

    def calculate_f(self):

        goal_matrix = np.ones((1,self.pos.shape[1])) * self.goal[0,0]
        for r in self.pos.shape[0] - 2:
            goal_matrix = np.hstack(goal_matrix,np.ones((1,self.pos.shape[1])) * self.goal[r + 1,0])


        f = self.acc - self.K * (goal_matrix - self.pos) + self.D * self.vel
        print(goal_matrix)

    def canonical_system_output(self):

        time = np.array(self.time_step * range(0, self.pos.shape[1]))
        return np.exp(time * self.canonical_constant)

