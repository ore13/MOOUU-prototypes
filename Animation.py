"""creates animations of Evolutionary algorithms for visualisation purposes"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import Test_suite as ts
import NSGA_II
import SPEA
import SPEA_2


class Animator:

    def __init__(self, fin_moo, pareto_front, savename):
        px, py = pareto_front
        self.fig, self.ax = plt.subplots()
        self.xdata, self.ydata = [], []
        plt.plot(px, py, 'b-')
        self.line, = plt.plot([], [], 'ro', animated=True)
        self.animation_generation = fin_moo.get_animation_points()
        ani = FuncAnimation(self.fig, self.update, frames=range(len(self.animation_generation)), init_func=self.init, blit=True)
        #plt.show()
        ani.save(savename)

    def init(self):
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 8)
        plt.xlabel("Objecive 1")
        plt.ylabel("Objective 2")
        plt.title("ZDT1 Test problem")
        return self.line,

    def update(self, frame):
        gen = self.animation_generation[frame]
        self.xdata, self.ydata = gen
        self.line.set_data(self.xdata, self.ydata)
        return self.line,

    def test(self):
        gen = self.animation_generation[0]
        self.xdata, self.ydata = gen
        self.ax.plot(self.xdata, self.ydata, 'o')
        plt.show()


if __name__ == "__main__":
    test = ts.TestAlgorithm(ts.ZDT1, SPEA_2.SPEA_2)
    moo_finished = test.moo
    pareto_front = test.problem.get_pareto_front()
    ani = Animator(moo_finished, pareto_front, 'SPEA2.mp4')
    #ani.test()

