"""Test problem in deb's book"""

import NSGA_II
import SPEA
import SPEA_2
import numpy as np
import matplotlib.pyplot as plt
from Test_suite import *


def run_test_problems():
    test = TestAlgorithm(ZDT1, NSGA_II.NSGA_II)
    test.new_plot("ZDT1", colour='b', marker='o')
    test.change_algorithm(NSGA_II.NSGA_II)
    test.replot(colour='g', marker='o')
    test.change_algorithm(SPEA_2.SPEA_2)
    test.replot(colour='r', marker='o')
    # test.save_plot('ZDT4', 'C:/Users/Violet/Desktop/GNS work/Comparisons')
    plt.show()


#run_test_problems()


def nice_plot_NSGA_II():
    # run NSGA_II
    objectives = [lambda x: abs(x[0]), lambda y: 1 / (abs(y[0]) + 0.1) + y[1] ** 2]
    bounds = [(-2, 2), (-2, 2)]
    moo = NSGA_II.NSGA_II(objectives, bounds, iterations=100, parent_pop_size=100)
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


#nice_plot_NSGA_II()


def nice_plot_SPEA():
    # run NSGA_II
    objectives = [lambda x: abs(x[0]), lambda y: 1 / (abs(y[0]) + 0.1) + y[1] ** 2]
    bounds = [(-2, 2), (-2, 2)]
    np.random.seed(1245678)
    moo = SPEA.SPEA(objectives, bounds, iterations=100, archive_size=30, population_size=70)
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
    plt.scatter(x, y, s=10, c='r', label='NSGA-II Pareto front approximation')
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.legend()
    plt.show()


#nice_plot_SPEA()


def nice_plot_SPEA_2():
    # run NSGA_II
    objectives = [lambda x: abs(x[0]), lambda y: 1 / (abs(y[0]) + 0.1) + y[1] ** 2]
    bounds = [(-2, 2), (-2, 2)]
    np.random.seed(1245678)
    moo = SPEA_2.SPEA_2(objectives, bounds, iterations=100, archive_size=50, population_size=50)
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
    plt.scatter(x, y, s=10, c='r', label='NSGA-II Pareto front approximation')
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.legend()
    plt.show()


# if __name__ == "__main__":
#     run_test_problems()
nice_plot_SPEA_2()