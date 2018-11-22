
"""Suite of test problems from ZDT - use with a plotter"""

import numpy as np
import matplotlib.pyplot as plt


class Test_Algorithm:
    """Set of ZDT test problems"""

    def __init__(self, problem, algorithm):
        self.problem = problem()
        objectives = [problem.f1, problem.f2]
        bounds = self.problem.bounds
        self.moo = algorithm(objectives, bounds, iterations=250, parent_pop_size=100)

    def plot_results(self):
        plt.figure("{}\n Solved using {}".format(str(self.problem), str(self.moo)))
        pareto_set = self.moo.run()
        x = []
        y = []
        for individual in pareto_set:
            x.append(self.problem.f1(individual.values))
            y.append(self.problem.f2(individual.values))
        plt.scatter(x, y, s=5, c='r', label='Pareto front approximation')
        px, py = self.problem.get_pareto_front()
        plt.plot(px, py, label='True Pareto Front')
        x_bounds = (int(np.rint(min(px))), int(np.rint(max(px))))
        plt.xlim(x_bounds)
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.legend()
        plt.title("{}\n Solved using {}".format(str(self.problem), str(self.moo)))


class Problem:
    """parent class for the set of ZDT test problems"""

    def __init__(self):
        self.bounds = []

    @staticmethod
    def f1(x):
        return np.NAN

    @staticmethod
    def f2(x, front=False):
        return np.NAN

    def get_pareto_front(self):
        x = np.array([np.linspace(self.bounds[0][0], self.bounds[0][1], 500)])
        return [self.f1(x), self.f2(x, front=True)]


class ZDT1(Problem):
    """represents ZDT problem 1"""

    def __init__(self):
        super().__init__()
        self.bounds = [(0, 1) for _ in range(30)]

    def __str__(self):
        return "Test problem ZDT1"

    @staticmethod
    def f1(x):
        return x[0]

    @staticmethod
    def f2(x, front=False):
        if front:
            g = 1
        else:
            g = 1 + 9 * np.sum(x[1:])
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
            g = 1 + 9 * np.sum(x[1:])
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
            g = 1 + 9 * np.sum(x[1:])
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
        try:
            result = g * (1 - np.sqrt(ZDT4.f1(x)/g))
        except RuntimeWarning:
            print(x)
        return result


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
            g = 1 + 9 * np.power(np.sum(x[1:]), 0.25)
        return g * (1 - np.power(ZDT6.f1(x)/g, 2))
