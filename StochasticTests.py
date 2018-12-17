"""Stochastic test functions. Testing constant, linear, and nonlinear distributions"""

import numpy as np
import scipy.stats as stat
import Test_suite as ts
import GeneticAlgorithms as ga
import matplotlib.pyplot as plt


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

    @staticmethod
    def calculate_objectives(d_vars, pars=np.array([0, 0]).reshape([2, 1])):
        """calculates the two objectives f1 and f2"""
        return np.array([StochasticTest.f1(d_vars, pars), StochasticTest.f2(d_vars, pars)])

    @staticmethod
    def number_parameters():
        return 2

    @staticmethod
    def number_decision_variables():
        return 2

    @staticmethod
    def number_objectives():
        return 2

    @staticmethod
    def parameter_means():
        return np.zeros(StochasticTest.number_decision_variables())

    @staticmethod
    def parameter_covariance():
        return np.diag([0.1, 0.1])

    @staticmethod
    def bounds():
        return [(0.1, 2), (0.1, 2)]


class UncertaintyPropagation:
    """
    class with both Ensemble and FOSM methods for calculating uncertainties
    """

    @staticmethod
    def parameters_from_gaussian(num_drawn, mean=np.array([0, 0]), cov=np.eye(2)):
        if len(mean) != len(cov):
            raise Exception("Standard deviation and error vectors must have same length")
        return stat.multivariate_normal.rvs(mean, cov, size=num_drawn).T

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
            eps_mat = eps * np.eye(model.number_objectives())
            mean_mat = np.tile(model.parameter_means(), (model.number_objectives(), 1))  # TODO check this again
            parameter_jacobian = model.calculate_objectives(d_vars, mean_mat + eps_mat) - \
                                 model.calculate_objectives(d_vars, mean_mat)
            parameter_jacobian /= eps
        objective_mean = model.calculate_objectives(d_vars, model.parameter_means())
        objective_sd = np.einsum('ij,ji->i', parameter_jacobian, np.matmul(model.parameter_covariance(),
                                                                           parameter_jacobian.T))
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
        return UncertaintyPropagation.risk_shifted_fosm(d_vars, risk=risk, model=StochasticTest)
    moo = ga.NSGA_II(uncertain_objective, StochasticTest.bounds(), StochasticTest.number_objectives(), iterations=50)
    pareto = moo.run()
    x = []
    y = []
    for individual in pareto:
        f1, f2 = individual.objective_values
        x.append(f1)
        y.append(f2)
    plt.plot(x, y, 'o', markersize=3, label='risk = {}'.format(risk))


if __name__ == "__main__":
    #Tests.test_risk_shift_ensemble()
    #Tests.test_risk_shift_fosm()
    example(0.5)
    example(0.1)
    example(0.9)
    plt.legend()
    plt.show()



