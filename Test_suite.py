
"""Suite of test problems from ZDT - use with a plotter"""

import numpy as np
import matplotlib.pyplot as plt


class TestAlgorithm:
    """Set of ZDT test problems"""

    def __init__(self, problem, algorithm):
        self.problem = problem()
        self.algorithm = algorithm
        bounds = self.problem.bounds
        self.moo = algorithm(self.problem.objectives, bounds, iterations=250)
        self.pareto_set, self.x, self.y = self.get_points()

    def change_problem(self, problem):
        self.problem = problem()
        bounds = self.problem.bounds
        self.moo = self.algorithm(self.problem.objectives, bounds, iterations=250)
        self.pareto_set, self.x, self.y = self.get_points()

    def change_algorithm(self, algorithm):
        self.algorithm = algorithm
        self.moo = algorithm(self.problem.objectives, self.problem.bounds, iterations=250)
        self.pareto_set, self.x, self.y = self.get_points()

    def new_plot(self, name, **kwargs):
        plt.close('all')
        plt.figure(name)
        plt.scatter(self.x, self.y, s=10, c=kwargs.get('colour'), marker=kwargs.get('marker'),
                    label='{}'.format(str(self.moo)))
        px, py = self.problem.get_pareto_front()
        plt.plot(px, py, label='True Pareto Front')
        x_bounds = (int(np.rint(min(px))), int(np.rint(max(px))))
        plt.xlim(x_bounds)
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.legend()
        plt.title(name)

    def replot(self, **kwargs):
        plt.scatter(self.x, self.y, s=10, c=kwargs.get('colour'), marker=kwargs.get('marker'),
                    label='{}'.format(str(self.moo)))
        plt.legend()

    def save_plot(self, name, directory):
        path = directory + '/' + name
        plt.savefig(fname=path, bbox_inches='tight')

    def get_points(self):
        pareto_set = np.array(self.moo.run())
        x = []
        y = []
        for individual in pareto_set:
            x.append(self.problem.f1(individual.values))
            y.append(self.problem.f2(individual.values))
        return pareto_set, x, y

    def convergence_metric(self):
        front = self.problem.get_pareto_front().T
        convergence = 0
        for individual in self.pareto_set:
            convergence += np.min(np.linalg.norm(individual.objective_values - front, 2, axis=1))
        return 1 / len(self.pareto_set) * convergence

    def distribution_metric(self, neighborhood):
        distribution = 0
        objective_matrix = np.zeros(shape=[len(self.pareto_set), len(self.pareto_set[0].objective_values)])
        for i, individual in enumerate(self.pareto_set):
            objective_matrix[i] = individual.objective_values
        for individual in self.pareto_set:
            diff_vector = individual.objective_values - objective_matrix
            distribution += np.count_nonzero(np.linalg.norm(diff_vector, 2, axis=1) < neighborhood)
        return 1 / (len(self.pareto_set) ** 2) * distribution

    def extent_metric(self):
        objective_matrix = np.zeros(shape=[len(self.pareto_set) ** 2, len(self.pareto_set[0].objective_values)])
        for i, individual in enumerate(self.pareto_set):
            for j, individual2 in enumerate(self.pareto_set):
                index = i * len(self.pareto_set) + j
                objective_matrix[index] = np.abs(individual.objective_values - individual2.objective_values)
        return np.sqrt(np.sum(np.max(objective_matrix, axis=0)))


class Problem:
    """parent class for the set of ZDT test problems"""

    def __init__(self):
        self.bounds = []
        self.objectives = [self.f1, self.f2]

    @staticmethod
    def f1(x):
        return np.NAN

    @staticmethod
    def f2(x, front=False):
        return np.NAN

    def get_pareto_front(self):
        x = np.array([np.linspace(self.bounds[0][0], self.bounds[0][1], 500)])
        return np.array([self.f1(x), self.f2(x, front=True)])


class ZDT1(Problem):
    """represents ZDT problem 1"""

    def __init__(self):
        super().__init__()
        self.bounds = [(0, 1) for _ in range(30)]

    def __str__(self):
        return "ZDT1"

    @staticmethod
    def f1(x):
        return x[0]

    @staticmethod
    def f2(x, front=False):
        if front:
            g = 1
        else:
            g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        return g * (1 - np.sqrt(ZDT1.f1(x) / g))


class ZDT2(Problem):
    """represents ZDT problem 2"""

    def __init__(self):
        super().__init__()
        self.bounds = [(0, 1) for _ in range(30)]

    def __str__(self):
        return "Test problem ZDT2"

    @staticmethod
    def f1(x):
        return x[0]

    @staticmethod
    def f2(x, front=False):
        if front:
            g = 1
        else:
            g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        return g * (1 - np.power(ZDT2.f1(x)/g, 2))


class ZDT3(Problem):
    """ZDT problem 3"""

    def __init__(self):
        super().__init__()
        self.bounds = [(0, 1) for _ in range(30)]

    def __str__(self):
        return "Test problem ZDT3"

    @staticmethod
    def f1(x):
        return x[0]

    @staticmethod
    def f2(x, front=False):
        if front:
            g = 1
        else:
            g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        return g * (1 - np.sqrt(ZDT3.f1(x)/g) - (ZDT3.f1(x)/g) * np.sin(10 * np.pi * ZDT3.f1(x)))


class ZDT4(Problem):
    """ZDT problem 4"""

    def __init__(self):
        super().__init__()
        self.bounds = [(0, 1) if i == 0 else (-5, 5) for i in range(10)]

    def __str__(self):
        return "Test problem ZDT4"

    @staticmethod
    def f1(x):
        return x[0]

    @staticmethod
    def f2(x, front=False):
        if front:
            g = 1
        else:
            g = 1 + 10 * (len(x) - 1) + np.sum(np.power(x[1:], 2) - 10 * np.cos(4 * np.pi * x[1:]))
        return g * (1 - np.sqrt(ZDT4.f1(x)/g))


class ZDT6(Problem):
    """ZDT problem 6"""

    def __init__(self):
        super().__init__()
        self.bounds = [(0, 1) for _ in range(10)]

    def __str__(self):
        return "Test problem ZDT6"

    @staticmethod
    def f1(x):
        return 1 - np.exp(-4 * x[0]) * np.power(np.sin(6 * np.pi * x[0]), 6)

    @staticmethod
    def f2(x, front=False):
        if front:
            g = 1
        else:
            g = 1 + 9 * np.power(np.sum(x[1:]) / (len(x) - 1), 0.25)
        return g * (1 - np.power(ZDT6.f1(x)/g, 2))
