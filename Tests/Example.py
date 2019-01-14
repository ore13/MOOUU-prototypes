"""Test problem in deb's book"""

from GeneticAlgorithms.NSGA_II import NSGA_II
from GeneticAlgorithms.SPEA import SPEA
from GeneticAlgorithms.SPEA_2 import SPEA_2
from TestProblems.Problem_suite import *
import numpy as np
import matplotlib.pyplot as plt


def run_test_problems():
    test = TestAlgorithm(ZDT1, NSGA_II)
    test.new_plot(str(ZDT1()), colour='b', marker='o')
    test.change_algorithm(SPEA)
    test.replot(colour='g', marker='o')
    test.change_algorithm(SPEA_2)
    test.replot(colour='r', marker='o')
    plt.show()


def constr_test():
    np.random.seed(390202)
    plt.figure()
    constr = TestAlgorithm(SRN, NSGA_II)
    constr.new_plot("SRN", colour='b', marker='o')
    plt.figure()
    constr.change_algorithm(SPEA)
    constr.replot(colour='g', marker='o')
    plt.figure()
    constr.change_algorithm(SPEA_2)
    constr.replot(colour='r', marker='o')
    #plt.show()


algorithms = {"nsga_ii": NSGA_II, "spea": SPEA, "spea_2": SPEA_2}


def nice_plot(algorithm_name):
    plt.figure()
    def objectives(x):
        return np.array([x[0], 1 / (x[0] + 0.1) + x[1] ** 2])
    if algorithm_name.lower() in algorithms.keys():
        algorithm = algorithms[algorithm_name.lower()]
    else:
        raise Exception("algorithm {} not found in known algorithms".format(algorithm_name.lower()))
    bounds = [(0, 4), (0, 4)]
    np.random.seed(1245678)
    moo = algorithm(objectives, bounds, 2)
    pareto_set = moo.run()
    # get data
    x = []
    y = []
    for individual in pareto_set:
        x_, y_ = objectives(individual.d_vars)
        x.append(x_)
        y.append(y_)
    # line representing real pareto set
    t = np.linspace(0, 4, 100)
    front = 1 / (t + 0.1)
    plt.plot(t, front, label='True Pareto front')
    plt.scatter(x, y, s=10, c='r', label='{} Pareto front approximation'.format(algorithm_name.upper()))
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.legend()
    #plt.show()


if __name__ == "__main__":
    constr_test()
    #run_test_problems()
    nice_plot('SPEA')
    nice_plot('NSGA_II')
    nice_plot('SPEA_2')
    plt.show()
