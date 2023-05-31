# -*- coding: utf-8 -*-
"""
Optimization methods for QDHMM

@author: nbhushan

"""

import random
import itertools
import numpy as np


def objective(taus, K, obs, weights):
    """Objective function

    The objective function is essentially the weighted average of
    the variance along the splits.

    Parameters
    -----------
    taus : list
        list of 'i' different tau configurations. Where each tau _i is of
        the form tau_ i = [0, t1 , t1+t2, ...]
    K : int
        The number of Simegy states. Equivalent to size(tau).
    obs : ndarray
        The observation sequence.
    weights : ndarray
        The posterior weights.

    Returns
    --------
    energy : list
        The value of the objective function for each tau configuration.

    """
    energy = []
    for tau in taus:
        mean = np.zeros((1, K))
        var = np.zeros((K, 1, 1))
        normalizer = np.zeros((K, weights.shape[1]))
        x = np.array(list(range(weights.shape[0])))
        state = np.digitize(x, tau, right=True)
        for k in range(0, K):
            normalizer[k, :] = (
                np.sum(weights[state == k], 0)
                / np.sum(np.sum(weights[state == k], 0)[np.newaxis, :], axis=1)[
                    :, np.newaxis
                ]
            )
            mean[:, k] = np.dot(normalizer[k, :], obs.T)
            obs_bar = obs - mean[:, k][:, np.newaxis]
            var[k, :, :] = np.dot(normalizer[k, :] * obs_bar, obs_bar.T)
        energy.append(
            (
                np.sum(np.sum(weights[0 : tau[0] + 1, :], 0)) * np.log(var[0, :, :])
                + np.sum(np.sum(weights[tau[0] + 1 : tau[1] + 1, :], 0))
                * np.log(var[1, :, :])
                + np.sum(np.sum(weights[tau[1] + 1 : tau[2] + 1, :], 0))
                * np.log(var[2, :, :])
                + np.sum(np.sum(weights[tau[2] + 1 :, :], 0)) * np.log(var[3, :, :])
            )[0][0]
        )
    return energy


def new_objective(tau, K, obs, weights):
    mean = np.zeros((1, K))
    var = np.zeros((K, 1, 1))
    normalizer = np.zeros((K, weights.shape[1]))
    x = np.array(list(range(weights.shape[0])))
    state = np.digitize(x, tau, right=True)
    for k in range(0, K):
        normalizer[k, :] = (
            np.sum(weights[state == k], 0)
            / np.sum(np.sum(weights[state == k], 0)[np.newaxis, :], axis=1)[
                :, np.newaxis
            ]
        )
        mean[:, k] = np.dot(normalizer[k, :], obs.T)
        obs_bar = obs - mean[:, k][:, np.newaxis]
        var[k, :, :] = np.dot(normalizer[k, :] * obs_bar, obs_bar.T)
    energy = (
        np.sum(np.sum(weights[0 : tau[0] + 1, :], 0)) * np.log(var[0, :, :])
        + np.sum(np.sum(weights[tau[0] + 1 : tau[1] + 1, :], 0)) * np.log(var[1, :, :])
        + np.sum(np.sum(weights[tau[1] + 1 : tau[2] + 1, :], 0)) * np.log(var[2, :, :])
        + np.sum(np.sum(weights[tau[2] + 1 :, :], 0)) * np.log(var[3, :, :])
    )[0][0]
    return energy


def generateneighbours(configuration, obs, steps, legal, weights, emmobj):
    """Generate neighbours for the current configuration.

    If legal, generate neighbours such that f(n) < f(c)

    Parameters
    -----------
    configuration : ndarray
        The tau parameter.
    obs : ndarray
        Sequence of observations.
    steps : list
        step to be taken in each dimension respectively.
    legal : bool
        Generate legal neighbours. Yes/No
    weights : ndarray
        posterior weights assigned to each state
    emmobj : QDHMM Emission model object
        QDHMM Emission model

    Returns
    -------
    list
        list of neighbours

    Notes
    -------
    This was during the initial exploration of optimization strategies.
    Will be deprecated.

    """
    tmp = []
    legalneighbours = []
    for idx, tau in enumerate(configuration[1:]):
        for step in steps:
            tmp.append(tau)
            tmp.append(tau + step)
            tmp.append(tau - step)
    a = itertools.combinations(tmp, configuration.shape[0] - 1)
    b = [list(item) for item in list(a)]
    [c.insert(0, 0) for c in b]
    possible_neighbours = b
    possible_neighbours[:] = [np.array(n) for n in possible_neighbours]
    possible_neighbours = checkpopulation(possible_neighbours, emmobj.D)
    if legal:
        f_c = objective([configuration], emmobj.K, obs, weights)[0]
        scores_n = objective(possible_neighbours, emmobj.K, obs, weights)
        for idx, f_n in enumerate(scores_n):
            if f_n <= f_c:
                legalneighbours.append(possible_neighbours[idx])
        return legalneighbours
    return possible_neighbours


def checkpopulation(population, D):
    """Remove illegal configuration by applying hard contraints on the
        time-outs.

    Parameters
    ----------
    population : list
        list of configurations
    D : int
        np.cumsum(tau)+2

    Returns
    -------
    list
        list of legal configurations which do not violate any of the constraints.

    Notes
    -------
    Constraints
    e.g. tau's must be monotonically increasing.
    tau = [0, x, y], then y!<x and x,y < D-1.

    """
    good_population = [
        n
        for n in population
        if not (n[1] >= n[2]) and n[1] >= 1 and (n >= 0).all() and (n < D - 1).all()
    ]
    return good_population


def localsearch(emmobj, obs, step, maxiter, weights):
    """Drive the local search.

    Drive the local search, start with infeasible solution,
    then move around in the local neighbourhood towards a
    feasible solution (local optima)

    Parameters
    -----------
    emmobj : QDHMM Emission model object
        QDHMM Emission model
    obs : ndarray
        Sequence of observations
    step : int
        The step to take in each dimension. Default = 1.
    maxiter : int
        The number of iterations to continue the search.
    weights : ndarray
        The posterior weights.

    Returns
    --------
    configuration : ndarray
        A (locally) optimal solution
    history : list
        A list of the configurations searched in the space. Used for
        visualization of the search

    Notes
    ------
    Note that local search will always return a solution even if interrupted.

    """
    # pdb.set_trace()
    history = []

    configuration = emmobj.tau
    history.append(configuration)
    c = 0
    neighbours = generateneighbours(configuration, obs, [step], True, weights, emmobj)
    while len(neighbours) > 0 and c < maxiter:
        c = c + 1
        configuration = random.choice(neighbours)
        history.append(configuration)
        neighbours = generateneighbours(
            configuration, obs, [step], True, weights, emmobj
        )
    return configuration, history


def generatepopulation(taus, weights, obs, number, step, D):
    """
    Generates a population of configurations for the genetic algorithm
    """
    population = []
    size = np.ceil(50 * np.exp(-number))
    np.random.seed(taus[1])
    step = np.ceil(step * np.exp(-number))
    a = np.random.randint(taus[1] - step, taus[1] + step, size)
    np.random.seed(taus[2])
    b = np.random.randint(taus[2] - step, taus[2] + step, size)
    c = []
    for a, b in zip(a, b):
        c.append(np.array([0, a, b]))
    population = [
        n
        for n in c
        if not (n[1] >= n[2]) and n[1] >= 1 and (n >= 0).all() and (n < D - 1).all()
    ]
    return population


def geneticalgorithm(obj, obs, maxiter, weights):
    count = 0
    configuration = obj.tau
    history = []
    history.append(configuration)
    step = np.ceil((np.min(np.diff(configuration))) / 2)
    while count < maxiter:
        step = np.ceil(step * np.exp(-count))
        old_energy = objective(obj.K, [configuration], obs, weights)[0]
        population = generatepopulation(configuration, weights, obs, count, step, obj.D)
        if len(population) > 1:
            fitness = objective(obj.K, population, obs, weights)
            strongest_tau1 = population[
                np.argsort(fitness)[np.random.randint(0, 2, 1)]
            ][1]
            strongest_tau2 = population[
                np.argsort(fitness)[np.random.randint(0, 2, 1)]
            ][2]
            newtau = np.array([0, strongest_tau1, strongest_tau2])
            new_energy = objective(obj.K, [newtau], obs, weights)[0]
            if old_energy >= new_energy:
                configuration = newtau
                history.append(configuration)
        count = count + 1
    # if (taus==configuration).all():
    #    taus = self.localsearch(configuration,  obs, 1, 10, weights)
    return configuration, history


def simulated_annealing(obj, weights, obs, maxiter):
    # pdb.set_trace()
    history = []
    configuration = obj.tau
    history.append(configuration)
    f_c = objective([configuration], obj.K, obs, weights)[0]
    bestconfig = configuration
    bestenergy = f_c
    count = 0
    t_init = 10000
    t_end = 0.01
    alpha = 0.9
    T = t_init
    naccept = 0
    # step=100
    step = np.diff(configuration) / 2
    while T > t_end:
        step = np.ceil(step * np.exp(-count))
        neighbours = generateneighbours(configuration, obs, step, False, weights, obj)
        fitness = objective(neighbours, obj.K, obs, weights)
        newconfiguration = neighbours[np.argmin(fitness)]
        f_n = np.min(fitness)
        delta = f_n - f_c
        if f_n < bestenergy or acceptconfiguration(delta, T) > random.random():
            naccept = naccept + 1
            if f_n < bestenergy:
                bestconfig = newconfiguration
                bestenergy = f_n
            configuration = newconfiguration
            history.append(configuration)
            f_c = f_n
        T = np.exp(-alpha) * T
        count = count + 1
    return bestconfig, history


def acceptconfiguration(delta, T):
    return np.exp(-delta / T)
