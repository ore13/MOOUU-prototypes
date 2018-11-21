"""Test problem in deb's book"""

import NSGA_II
import numpy as np
import matplotlib.pyplot as plt


def nice_plot():
    # run NSGA_II
    np.random.seed(123426)
    objectives = [lambda x: abs(x[0]), lambda y: 1 / (abs(y[0]) + 0.1) + y[1] ** 2]
    bounds = [(-2, 2), (-2, 2)]
    moo = NSGA_II.NSGA_II(objectives, bounds, iterations=10, parent_pop_size=100)
    pareto_set = moo.run()
    # get data
    x = []
    y = []
    for individual in pareto_set:
        x.append(objectives[0](individual.values))
        y.append(objectives[1](individual.values))
    # line representing real pareto set
    t = np.linspace(0, 2, 100)
    f = lambda z: 1 / (abs(z) + 0.1)
    front = f(t)
    # create plot
    plt.plot(t, front, label='True Pareto front')
    plt.scatter(x, y, s=5, c='r', label='NSGA-II Pareto front approximation')
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.legend()
    plt.show()


nice_plot()



