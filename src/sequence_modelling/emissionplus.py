# -*- coding: utf-8 -*-
"""
Emission distributions for QDHMM

@author: nbhushan

"""

import time
import pdb

import numpy as np
from scipy.stats import norm

import sequence_modelling.DiscreteOptim as optim
import sequence_modelling.hmmviz as viz


class Gaussian:
    """The Gaussian emission model for a QDHMM.

    Attributes
    ----------
    mu : ndarray
            mean, 'mu' is defined by  ndarray of shape [d, K]. Where
            d is the dimension of the features and K is the total number of
            states.
    var : ndarray
            variance,
            'var' is defined by  ndarray of shape [K, d, d]. Where
            d is the dimension of the features and K is the total number of
            states. Note: borrowed from standard HMM 'covar'.
    tau : ndarray
        The time-out parameters. Note: if the individual timeouts
        are t1,t2, t3.. then tau = [0, t1, t1+t2, t1+t2+t3, .. ]

    Notes
    -----
    D is the cumulative sum of the individual timeouts + 2.

    """

    def __init__(self, mu, var, tau):
        self.mu = mu
        self.var = var
        self.K = mu.shape[1]
        self.dim = mu.shape[0]
        self.tau = tau
        self.D = (np.max(tau) * 2) + 2
        # self.D = 360

    def __repr__(self):
        i = "Gaussian Emissions\n"
        tau = "" + "tau's: \n %s \n" % (str(self.tau))
        mu = "" + "Emmission Mean: \n %s \n" % (str(self.mu))
        var = "" + "covar:\n %s" % (str(self.var))
        return i + tau + mu + var

    def Sample(self, zes):
        """Generates a Gaussian observation sequence from a given state sequence.


        Parameters
        -----------
        stateseq : ndarray
            The state sequence which is used to generate the observation
            sequence.

        Returns
        --------
        ndarray
            Observation sequence.

        Notes
        -------
        The observation sequence can only be univariate Gaussian in the
        QDHMM Emission model.

        """
        state = np.digitize(zes, self.tau, right=True)
        if self.dim == 1:
            return np.random.normal(
                self.mu[:, state].flatten(), np.sqrt(self.var[state, :, :].flatten())
            )
        else:
            return np.random.multivariate_normal(
                self.mu[:, state], np.sqrt(self.var[state, :, :])
            )

    def Fit(self, obs, weights, estimatetau, metaheuristic):
        """Fit a Gaussian to the state distributions after observing the data.

        Parameters
        -----------
        obs : ndarray
            Observation sequence.
        weights : ndarray
            The weights attached to each state (posterior distribution).
        estimatetau : bool
            Find optimal tau's Yes/No
        metaheuristic : {'local', 'sa', 'genetic'}, optional
            The meta-heuristic to be used to solve the objective in the M-step.
            'local' is simple local search. 'genetic' is genetic algorithm and
            'sa' is simulated annealing.

        Notes
        --------
        Estimation of the tau parameters is done every alternate iteration
        of the QDHMM EM algorithm.
        Call viz.plotcontour only if you wish to view the search propogation.
        Recommended for search in a small space.


        """
        if estimatetau:
            if metaheuristic == "genetic":
                t = time.time()
                genetic, history = optim.geneticalgorithm(self, obs, 20, weights)
                print(("tau: ", genetic))
                print(("genetic search time:", time.time() - t))
                self.tau = genetic
            elif metaheuristic == "local":
                t = time.time()
                local, history = optim.localsearch(self, obs, 1, 20, weights)
                print(("tau: ", local))
                print(("local search time:", time.time() - t))
                self.tau = local
            # elif metaheuristic=='sa':
            #     t=time.time()
            #     aneal, history= optim.simulated_annealing(self, weights, obs, 20)
            #     print(('aneal search time:', time.time()-t))
            #     print(('tau: ',aneal))
            #     print('Scipy anneal')
            #     pdb.set_trace()
            #     print((anneal(optim.new_objective, self.tau, (self.K, obs, weights), schedule='cauchy', T0=10000, Tf=0.0001)))
            #     self.tau = aneal
            # viz.plotcontour(self.K, self.tau, weights, obs, history, str(metaheuristic+str(iteration)))
        normalizer = np.zeros((self.K, weights.shape[1]))
        x = np.array(list(range(weights.shape[0])))
        state = np.digitize(x, self.tau, right=True)
        for k in range(self.K):
            normalizer[k, :] = (
                np.sum(weights[state == k], 0)
                / np.sum(np.sum(weights[state == k], 0)[np.newaxis, :], axis=1)[
                    :, np.newaxis
                ]
            )
            self.mu[:, k] = np.dot(normalizer[k, :], obs.T)
            obs_bar = obs - self.mu[:, k][:, np.newaxis]
            self.var[k, :, :] = np.dot(normalizer[k, :] * obs_bar, obs_bar.T)

    def likelihood(self, obs):
        """To compute likelihood of drawing an observation 'y' from a
            given state:  P(x | Z_t) = N(mu,var).

        Parameters
        -----------
        obs : ndarray
            The observation sequence

        Returns
        --------
        logB : ndarray
            THe observation probability distribution.

        """
        # pdb.set_trace()
        B = np.zeros((self.D, obs.shape[1]))
        state = np.array((self.D))
        d = list(range(self.D))
        state = np.digitize(d, self.tau, right=True)
        for k in range(self.K):
            B[state == k, :] = norm.pdf(
                obs, loc=self.mu[:, k], scale=np.sqrt(self.var[k, :, :])
            )
        return B
