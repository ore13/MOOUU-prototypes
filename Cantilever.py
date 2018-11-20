"""Test problem in deb's book"""

import NSGA_II
import numpy as np
import matplotlib.pyplot as plt


def testfunc():
    plt.close(1)
    objective1 = lambda x : abs(x[0])
    objective2 = lambda x: 1 / (abs(x[0]) + 0.1) + x[1] ** 2
    F = [objective1, objective2]
    bounds = [(-2, 2), (-2, 2)]
    moo = NSGA_II.NSGA_II(F, bounds, iterations=10)
    pareto_set = moo.run()
    x = []
    y = []
    for pop in pareto_set:
        x.append(objective1(pop.values))
        y.append(objective2(pop.values))
    real_pareto = lambda t: 1 / (abs(t) + 0.1)
    t_space = np.linspace(0, 15, 100)
    plt.plot(t_space, real_pareto(t_space))
    plt.scatter(x, y, s=5, c='r')
    plt.show()

def visualisePopulation():
    plt.close(1)
    objective1 = lambda x : abs(x[0])
    objective2 = lambda x: 1 / (abs(x[0]) + 0.1) + x[1] ** 2
    F = [objective1, objective2]
    bounds = [(-2, 2), (-2, 2)]
    moo = NSGA_II.NSGA_II(F, bounds, iterations=10)
    pareto_set = moo.run()
    populations = moo.pop_storage

testfunc()



