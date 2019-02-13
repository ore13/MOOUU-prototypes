"""Stochastic test functions. Testing constant, linear, and nonlinear distributions"""

import numpy as np
import scipy.stats as stat
import ga
import matplotlib.pyplot as plt
import time
import random
from TestProblems.Problem_suite import *
import pandas as pd


def additive_parameter_interaction(d_vars, pars):
    if len(d_vars) != pars.shape[1]:
        raise Exception('Should have same number of parameters and decision variables')
    if pars.shape[1] % 2 == 0:
        even = np.arange(pars.shape[1] // 2) * 2
        odd = even + 1
    else:
        even = np.arange(pars.shape[1] // 2 + 1) * 2
        odd = np.arange(pars.shape[1] // 2) * 2 + 1
    return 1 / pars.shape[1] * np.array([np.sum(pars[:, even], axis=1), np.sum(pars[:, odd], axis=1)]).T


def multiplicitive_parameter_interaction(d_vars, pars):
    if len(d_vars) != pars.shape[1]:
        raise Exception('Should have same number of parameters and decision variables')
    if pars.shape[1] % 2 == 0:
        even = np.arange(pars.shape[1] // 2 - 1) * 2 + 2
        odd = even + 1
    else:
        even = np.arange(pars.shape[1] // 2) * 2 + 2
        odd = np.arange(pars.shape[1] // 2) * 2 + 1
    return np.array([d_vars[0] * pars[:, 0] + np.sum(d_vars[even] * pars[:, even], axis=1),
                     np.sum(d_vars[odd] * pars[:, odd], axis=1)]).T


def nonlinear_parameter_interaction(d_vars, pars):
    if len(d_vars) != len(pars):
        raise Exception('Should have same number of parameters and decision variables')
    if len(pars) % 2 == 0:
        even = np.arange(len(pars) // 2 - 1) * 2 + 2
        odd = even + 1
    else:
        even = np.arange(len(pars) // 2) * 2 + 2
        odd = np.arange(len(pars) // 2) * 2 + 1
    f1 = np.sinh(5 * (d_vars[0] + 1) * pars[0]) + np.sum(np.sinh(5 * d_vars[even] * pars[even]))
    f2 = np.sum(np.sinh(5 * d_vars[odd] * pars[odd]))
    return np.array([f1, f2])


class UncertaintyPropagation:
    """
    class with both Ensemble and FOSM methods for calculating uncertainties
    """

    @staticmethod
    def parameters_from_gaussian(num_drawn, mean=np.array([0, 0]), cov=np.eye(2)):
        if len(mean) != len(cov):
            raise Exception("Standard deviation and error vectors must have same length")
        rvs = stat.multivariate_normal.rvs(mean, cov, size=num_drawn)
        return rvs

    @staticmethod
    def risk_shifted_ensemble(d_vars, risk, eval_func, comb, pars):
        d_vars_copy = d_vars.T
        evals = pars.shape[0]
        index = int(risk * evals)
        objectives = []
        for i, d_var in enumerate(d_vars_copy):
            eval_ensemble = comb(d_var, pars).T
            obs_ensemble = eval_func(eval_ensemble)
            obs_ensemble.sort(axis=1)
            objectives.append(obs_ensemble[:, index])
        return np.array(objectives).T
        #return eval_func(d_vars)

    @staticmethod
    def pyemu_interactions_ensemble(d_vars, pars, interaction, model, risk):
        d_vars_copy = d_vars.T
        evals = pars.shape[0]
        index = int(risk * evals)
        objectives = []
        normal = model(d_vars)
        for objective, dvar in zip(normal.T, d_vars_copy):
            obs_ensemble = objective + interaction(dvar, pars)
            obs_ensemble.sort(axis=0)
            objectives.append(obs_ensemble[index, :])
        return np.array(objectives).T

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


def additive(dvar, pars):
    assert len(dvar) == pars.shape[1]
    par_copy = pars.copy()
    par_copy[:, 1:] = 1/10 * par_copy[:, 1:]
    return dvar + par_copy


def ZDT_risk_vals(problem, risk):
    mean = np.zeros(shape=problem.number_decision_vars())
    cov = 0.2 * np.eye(problem.number_decision_vars())
    par_ensemble = UncertaintyPropagation.parameters_from_gaussian(400, mean, cov)
    # mean = par_ensemble.mean(axis=0)
    # for col, mean in zip(np.arange(0, 30), mean):
    #     par_ensemble[:, col] = par_ensemble[:, col] - mean  # centering to reduce bias
    if np.any(par_ensemble > 1) or np.any(par_ensemble < -1):
        where_above = np.where(par_ensemble > 1)
        where_below = np.where(par_ensemble < -1)
        par_ensemble[where_above] = 1
        par_ensemble[where_below] = -1

    def uncertain_objective(d_vars):
        obj = UncertaintyPropagation.pyemu_interactions_ensemble(d_vars, par_ensemble, multiplicitive_parameter_interaction,
                                                                 model=problem.objective_vector, risk=risk)
        return obj.T
    moo = ga.NSGA_II(uncertain_objective, problem().bounds, problem.number_objectives(), iterations=500,
                     parent_pop_size=200)
    pareto = moo.run()
    x = []
    y = []
    for individual in pareto:
        f1, f2 = individual.objective_values
        x.append(f1)
        y.append(f2)
    data = [individual.d_vars for individual in pareto]
    df = pd.DataFrame(data=data, columns=['dvar{}'.format(i) for i in range(30)])
    df.to_csv('Opt_dvars_risk_{:2f}.csv'.format(risk))
    plt.plot(x, y, 'o', markersize=3, label='risk = {}'.format(risk))


if __name__ == "__main__":
    problem = ZDT3
    ZDT_risk_vals(problem, 0.5)
    ZDT_risk_vals(problem, 0.7)
    ZDT_risk_vals(problem, 0.9)
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.legend()
    plt.show()




