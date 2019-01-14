"""Stochastic test functions. Testing constant, linear, and nonlinear distributions"""

import numpy as np
import scipy.stats as stat
import ga
import matplotlib.pyplot as plt
import time


class StochasticTest:

    def __init__(self):
        raise Exception("Class should not be instatiated")

    @staticmethod
    def f1(d_vars, pars=np.array([0, 0]).reshape([2, 1])):
        """
        Represents a model giving an objective function as it's output
        :param d_vars: decision vector
        :param pars: array of parameter combination vectors
        :return: output of model given parameters. length of output is equal to number of supplied parameters
        """
        return d_vars[0] + pars[0]

    @staticmethod
    def f2(d_vars, pars=np.array([0, 0]).reshape([2, 1])):
        """
        Represents a model with an objective function as it's output
        :param d_vars: decision vector
        :param pars: parameters of the model, which should be taken from a normal distribution
        :return: output of model given parameters
        """
        return 1 / d_vars[0] + d_vars[1] + pars[1]

    @classmethod
    def calculate_objectives(cls, d_vars, pars=np.array([0, 0]).reshape([2, 1])):
        """calculates the two objectives f1 and f2"""
        return np.array([cls.f1(d_vars, pars), cls.f2(d_vars, pars)])

    @staticmethod
    def number_parameters():
        return 2

    @staticmethod
    def number_decision_variables():
        return 2

    @staticmethod
    def number_objectives():
        return 2

    @classmethod
    def parameter_means(cls):
        return np.zeros(cls.number_parameters())

    @staticmethod
    def parameter_covariance():
        return np.diag([1, 1])

    @staticmethod
    def bounds():
        return [(0.1, 2), (0.1, 2)]

    @staticmethod
    def _quadratic_positive_root(a, b, c):
        return (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


class StochasticParaboloid(StochasticTest):

    def __init__(self):
        super().__init__()

    @staticmethod
    def f1(d_vars, pars=np.array([[1]])):
        return (d_vars[0] - 2) ** 2 + d_vars[1] ** 2 + 0.1 * (d_vars[0] + pars[0]) ** 3

    @staticmethod
    def f2(d_vars, pars=np.array([[1]])):
        return (d_vars[0] + 2) ** 2 + d_vars[1] ** 2 - 0.1 * (d_vars[0] + pars[0]) ** 3

    @staticmethod
    def number_parameters():
        return 1

    @staticmethod
    def bounds():
        return [(-5, 5), (-5, 5)]

    @staticmethod
    def parameter_covariance():
        return np.diag([1])

    @staticmethod
    def pareto_front(risk):
        """:returns pareto front given a certain level of risk"""
        beta_up = stat.norm.ppf(risk)
        beta_down = stat.norm.ppf(1 - risk)
        a, b, c = (3 * 0.1, 2 + 6 * 0.1 * beta_up, 3 * 0.1 * beta_up ** 2 - 4)
        argminf1 = StochasticTest._quadratic_positive_root(a, b, c)
        a, b, c = (-3 * 0.1, 2 - 6 * 0.1 * beta_down, 4 - 3 * 0.1 * beta_down ** 2)
        argminf2 = StochasticTest._quadratic_positive_root(a, b, c)
        pareto_set = np.vstack((np.linspace(argminf2, argminf1), np.zeros(50)))
        f1 = StochasticParaboloid.f1(pareto_set, [beta_up])
        f2 = StochasticParaboloid.f2(pareto_set, [beta_down])
        return np.vstack((f1, f2))


class StochasticParaboloid2(StochasticParaboloid):

    def __init__(self):
        super().__init__()

    @staticmethod
    def f1(d_vars, pars=np.array([[1]])):
        return (d_vars[0] - 2) ** 2 + d_vars[1] ** 2 + 0.05 * (d_vars[1] + pars[0]) ** 3

    @staticmethod
    def f2(d_vars, pars=np.array([[1]])):
        return (d_vars[0] + 2) ** 2 + d_vars[1] ** 2 + 0.05 * (d_vars[1] + pars[0]) ** 3

    @staticmethod
    def pareto_front(risk):
        beta_up = stat.norm.ppf(risk)
        arg_min_f1 = StochasticTest._quadratic_positive_root(0.15, 2 - 0.3 * beta_up, 0.15 * beta_up ** 2)
        pareto_set = np.vstack((np.linspace(-2, 2), np.full(50, arg_min_f1)))
        f1 = StochasticParaboloid2.f1(pareto_set, [beta_up])
        f2 = StochasticParaboloid2.f2(pareto_set, [beta_up])
        return np.vstack((f1, f2))


class UncertaintyPropagation:
    """
    class with both Ensemble and FOSM methods for calculating uncertainties
    """

    @staticmethod
    def parameters_from_gaussian(num_drawn, mean=np.array([0, 0]), cov=np.eye(2)):
        if len(mean) != len(cov):
            raise Exception("Standard deviation and error vectors must have same length")
        rvs = stat.multivariate_normal.rvs(mean, cov, size=num_drawn)
        return rvs.reshape((len(mean), num_drawn))

    @staticmethod
    def risk_shifted_ensemble(d_vars, risk, model, ensemble=None, evals=30):
        if ensemble is None:
            mean = model.parameter_means()
            cov = model.parameter_covariance()
            pars = UncertaintyPropagation.parameters_from_gaussian(evals, mean, cov)
            ensemble = model.calculate_objectives(d_vars, pars)
        else:
            evals = ensemble.shape[0]
        ensemble.sort()
        index = int(risk * evals)
        return ensemble[:, index]

    @staticmethod
    def risk_shifted_fosm(d_vars, risk, model, parameter_jacobian=None):
        if parameter_jacobian is None:
            eps = 1e-10
            eps_mat = eps * np.eye(model.number_parameters())
            mean_mat = np.tile(model.parameter_means(), (model.number_parameters(), 1))  # TODO check this again
            parameter_jacobian = model.calculate_objectives(d_vars, mean_mat + eps_mat) - \
                                 model.calculate_objectives(d_vars, mean_mat)
            parameter_jacobian /= eps
        objective_mean = model.calculate_objectives(d_vars, model.parameter_means())
        objective_sd = np.sqrt(np.einsum('ij,ji->i', parameter_jacobian, np.matmul(model.parameter_covariance(),
                                                                           parameter_jacobian.T)))
        return objective_sd * stat.norm.ppf(risk) + objective_mean


class Tests:

    @staticmethod
    def test_risk_shift_ensemble():
        np.random.seed(129302)
        risk_shifted = UncertaintyPropagation.risk_shifted_ensemble([1, 1], 0.5, StochasticTest, evals=2000)
        assert np.all(np.isclose(risk_shifted, [1, 2], atol=1e-1))
        risk_shifted = UncertaintyPropagation.risk_shifted_ensemble([1, 1], 0.9, StochasticTest, evals=2000)
        assert np.all(np.isclose(risk_shifted, [2.28155, 3.28155], atol=1e-1))
        risk_shifted = UncertaintyPropagation.risk_shifted_ensemble([1, 1], 0.1, StochasticTest, evals=2000)
        assert np.all(np.isclose(risk_shifted, [-0.28155156, 0.71844843], atol=1e-1))

    @staticmethod
    def test_risk_shift_fosm():
        risk_shifted = UncertaintyPropagation.risk_shifted_fosm([1, 1], 0.5, StochasticTest)
        assert np.all(np.isclose(risk_shifted, [1, 2], atol=1e-1))
        risk_shifted = UncertaintyPropagation.risk_shifted_fosm([1, 1], 0.9, StochasticTest)
        assert np.all(np.isclose(risk_shifted, [2.28155, 3.28155], atol=1e-1))
        risk_shifted = UncertaintyPropagation.risk_shifted_fosm([1, 1], 0.1, StochasticTest)
        assert np.all(np.isclose(risk_shifted, [-0.28155156, 0.71844843], atol=1e-1))


def example(risk):
    def uncertain_objective(d_vars):
        return UncertaintyPropagation.risk_shifted_ensemble(d_vars, risk=risk, model=StochasticTest)
    moo = ga.NSGA_II(uncertain_objective, StochasticTest.bounds(), StochasticTest.number_objectives(), iterations=50)
    pareto = moo.run()
    x = []
    y = []
    for individual in pareto:
        f1, f2 = individual.objective_values
        x.append(f1)
        y.append(f2)
    plt.plot(x, y, 'o', markersize=3, label='risk = {}'.format(risk))


def risk_affecting_dec_space_example(risk, method, model):
    def uncertain_objective(d_vars):
        if method.lower() == 'fosm':
            return UncertaintyPropagation.risk_shifted_fosm(d_vars, risk=risk, model=model)
        elif method.lower() == 'ensemble':
            return UncertaintyPropagation.risk_shifted_ensemble(d_vars, risk=risk, model=model, evals=100)
        else:
            raise Exception("no known method {}".format(method))
    moo = ga.NSGA_II(uncertain_objective, model.bounds(), model.number_objectives(), iterations=100)
    pareto = moo.run()
    x = []
    y = []
    for individual in pareto:
        f1, f2 = individual.objective_values
        x.append(f1)
        y.append(f2)
    plt.plot(x, y, 'o', markersize=3, label='risk = {}'.format(risk))
    x, y = model.pareto_front(risk)
    plt.plot(x, y, label='True Pareto front, risk = {}'.format(risk))


if __name__ == "__main__":
    Tests.test_risk_shift_ensemble()
    Tests.test_risk_shift_fosm()
    risk_affecting_dec_space_example(0.05, 'ensemble', StochasticParaboloid)
    risk_affecting_dec_space_example(0.5, 'ensemble', StochasticParaboloid)
    risk_affecting_dec_space_example(0.95, 'ensemble', StochasticParaboloid)
    plt.title("MOOUU test problem StochasticParaboloid\nusing ensemble method")
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.legend()
    plt.figure(2)
    risk_affecting_dec_space_example(0.01, 'fosm', StochasticParaboloid)
    risk_affecting_dec_space_example(0.5, 'fosm', StochasticParaboloid)
    risk_affecting_dec_space_example(0.99, 'fosm', StochasticParaboloid)
    plt.title("MOOUU test problem StochasticParaboloid\nusing FOSM method")
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.legend()
    plt.show()




