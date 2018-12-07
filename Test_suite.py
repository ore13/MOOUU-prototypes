
"""Suite of test problems from ZDT - use with a plotter"""

import numpy as np
import matplotlib.pyplot as plt


class TestAlgorithm:
    """Set of ZDT test problems"""

    def __init__(self, problem, algorithm):
        self.problem = problem()
        self.algorithm = algorithm
        bounds = self.problem.bounds
        if self.problem.constrained():
            constraints = self.problem.constraints
        else:
            constraints = None
        self.moo = algorithm(self.problem.objectives, bounds, iterations=500, constraints=constraints)
        self.pareto_set, self.x, self.y = self.get_points()

    def change_problem(self, problem):
        self.problem = problem()
        bounds = self.problem.bounds
        if self.problem.constrained():
            constraints = self.problem.constraints
        else:
            constraints = None
        self.moo = self.algorithm(self.problem.objectives, bounds, iterations=500, constraints=constraints)
        self.pareto_set, self.x, self.y = self.get_points()

    def change_algorithm(self, algorithm):
        self.algorithm = algorithm
        if self.problem.constrained():
            constraints = self.problem.constraints
        else:
            constraints = None
        self.moo = algorithm(self.problem.objectives, self.problem.bounds, iterations=500, constraints=constraints)
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
            x.append(self.problem.f1(individual.d_vars))
            y.append(self.problem.f2(individual.d_vars))
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
        self.number_of_decision_vars = None
        if self.constrained():
            self.constraints = [self.constraint1, self.constraint2]

    @staticmethod
    def f1(x):
        return np.NAN

    @staticmethod
    def f2(x, front=False):
        return np.NAN

    def get_pareto_front(self):
        x = np.array([np.linspace(self.bounds[0][0], self.bounds[0][1], 500)])
        return np.array([self.f1(x), self.f2(x, front=True)])

    @staticmethod
    def number_decision_vars():
        return None

    def constrained(self):
        return False

    @staticmethod
    def constraint1(x):
        return np.NAN

    @staticmethod
    def constraint2(x):
        return np.NAN


class ZDT1(Problem):
    """represents ZDT problem 1"""

    def __init__(self):
        super().__init__()
        self.number_of_decision_vars = 30
        self.bounds = [(0, 1) for _ in range(self.number_of_decision_vars)]

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

    @staticmethod
    def number_decision_vars():
        return 30


class ZDT2(Problem):
    """represents ZDT problem 2"""

    def __init__(self):
        super().__init__()
        self.number_of_decision_vars = 30
        self.bounds = [(0, 1) for _ in range(self.number_of_decision_vars)]

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

    @staticmethod
    def number_decision_vars():
        return 30


class ZDT3(Problem):
    """ZDT problem 3"""

    def __init__(self):
        super().__init__()
        self.number_of_decision_vars = 30
        self.bounds = [(0, 1) for _ in range(self.number_of_decision_vars)]

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

    @staticmethod
    def number_decision_vars():
        return 30


class ZDT4(Problem):
    """ZDT problem 4"""

    def __init__(self):
        super().__init__()
        self.number_of_decision_vars = 10
        self.bounds = [(0, 1) if i == 0 else (-5, 5) for i in range(self.number_of_decision_vars)]

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

    @staticmethod
    def number_decision_vars():
        return 10


class ZDT6(Problem):
    """ZDT problem 6"""

    def __init__(self):
        super().__init__()
        self.number_of_decision_vars = 10
        self.bounds = [(0, 1) for _ in range(self.number_of_decision_vars)]

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

    @staticmethod
    def number_decision_vars():
        return 10


class CONSTR(Problem):
    """CONSTR problem for testing constraint handling"""

    def __init__(self):
        super().__init__()
        self.number_decision_vars = 2
        self.bounds = [(0.1, 1), (0, 5)]

    def __str__(self):
        return "Test problem CONSTR"

    def constrained(self):
        return True

    @staticmethod
    def constraint1(x):
        return (x[1] + 9 * x[0])/6 - 1

    @staticmethod
    def constraint2(x):
        return -x[1] + 9 * x[0] - 1

    @staticmethod
    def f1(x):
        return x[0]

    @staticmethod
    def f2(x, front=False):
        if front:
            raise Exception("This problem does not support the front flag")
        return (1 + x[1])/x[0]

    def get_pareto_front(self):
        x1_1 = np.linspace(7 / 18, 2 / 3)
        x1_2 = np.linspace(2 / 3, 1)
        f1 = np.concatenate((x1_1, x1_2))
        f2 = np.concatenate(((7 - 9 * x1_1) / x1_1, 1 / x1_2))
        return f1, f2


class SRN(Problem):
    """CONSTR problem for testing constraint handling"""

    def __init__(self):
        super().__init__()
        self.number_decision_vars = 2
        self.bounds = [(-20, 20), (-20, 20)]

    def __str__(self):
        return "Test problem CONSTR"

    def constrained(self):
        return True

    @staticmethod
    def constraint1(x):
        return np.power(x[0], 2) + np.power(x[1], 2) - 225

    @staticmethod
    def constraint2(x):
        return x[0] - 3 * x[1] + 10

    @staticmethod
    def f1(x):
        return np.power(x[0] - 2, 2) + np.power(x[1] - 1, 2) + 2

    @staticmethod
    def f2(x, front=False):
        if front:
            raise Exception("This problem does not support the front flag")
        return 9 * x[0] - np.power(x[1] - 1, 2)

    def get_pareto_front(self):
        maximiser1 = -2.5
        upper_bound = 1.1
        lowerx1 = np.full(100, maximiser1)
        lowerx2 = np.linspace((maximiser1 + 10)/3, np.sqrt(225 - maximiser1 ** 2), 100)
        upperx1 = np.linspace(maximiser1, upper_bound, 100)
        upperx2 = (10 + upperx1)/3
        x1 = np.concatenate((lowerx1, upperx1))
        x2 = np.concatenate((lowerx2, upperx2))
        return SRN.f1(np.array([x1, x2])), SRN.f2(np.array([x1, x2]))
