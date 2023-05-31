# -*- coding: utf-8 -*-
"""
Emission distributions for HMM

@author: nbhushan

"""

import numpy as np
from sequence_modelling.utils import logsumexp
from scipy.stats import norm


class Gaussian:
    """The Gaussian emission model for a standard HMM.

    Attributes
    ----------
    mu : ndarray
            mean, 'mu' is defined by  ndarray of shape [d, K]. Where
            d is the dimension of the features and K is the total number of
            states.
    covar : ndarray
            co-variance (variance),
            'covar' is defined by  ndarray of shape [K, d, d]. Where
            d is the dimension of the features and K is the total number of
            states.

    Notes
    -----
    In the case of univariate data, the covariance decomposes into
    the variance in the computations.

    """

    def __init__(self, mu, covar):
        self.mu = mu
        self.covar = covar
        self.K = mu.shape[1]
        self.dim = mu.shape[0]

    def __repr__(self):
        i = "Gaussian Emissions\n"
        mu = "" + "Emmission Mean: \n %s \n" % (str(self.mu))
        covar = "" + "covar:\n %s" % (str(self.covar))
        return i + mu + covar

    def sample(self, stateseq):
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
        The observation sequence can be univariate or multivariate Gaussian
        depending on the 'dim' parameter of the emission model.

        """
        if self.dim == 1:
            return np.random.normal(
                self.mu[:, stateseq].flatten(),
                np.sqrt(self.covar[stateseq, :, :]).flatten(),
            )
        else:
            return np.random.multivariate_normal(
                self.mu[:, stateseq].flatten(), self.covar[stateseq, :, :].flatten()
            )

    def fit(self, obs, logweights):
        """Fit a Gaussian to the state distributions after observing the data.

        Parameters
        -----------
        obs : ndarray
            Observation sequence.
        logweights : ndarray
            The weights attached to each state (posterior distribution).
            In log-space.


        """
        # oldmeans = self.mu.copy()
        logGamma = np.concatenate(logweights, 1)
        normalizer = np.exp(logGamma - logsumexp(logGamma, 1)[:, np.newaxis])
        for k in range(self.K):
            self.mu[:, k] = np.dot(normalizer[k, :][np.newaxis, :], obs.T)
            obs_bar = obs - self.mu[:, k][:, np.newaxis]
            self.covar[k, :, :] = np.dot(obs_bar * normalizer[k, :], obs_bar.T)

    def loglikelihood(self, obs):
        """To compute loglikelihood of drawing an observation 'y' from a
            given state:  log(P(x | Z_t)) = N(mu,covar).

        Parameters
        -----------
        obs : ndarray
            The observation sequence

        Returns
        --------
        logB : ndarray
            The observation probability distribution in log-space.

        """
        logB = np.zeros((self.K, obs.shape[1]))
        for k in range(self.K):
            logB[k, :] = norm.logpdf(
                obs, loc=self.mu[:, k], scale=np.sqrt(self.covar[k, :, :])
            )
        return logB

    def likelihood(self, y, state):
        """To compute likelihood of drawing an observation 'y' from a
            given state:  P(x | Z_t) = N(mu,covar).

        Parameters
        -----------
        obs : ndarray
            The observation sequence

        Returns
        --------
        logB : ndarray
            THe observation probability distribution.

        """
        return np.exp(self.loglikelihood(y, state))


class Discrete:
    """The Discrete emission model for a standard HMM.

    Attributes
    ----------
    pvector : ndarray
            The discrete observation distribution.
    cvector : ndarray
            The allowed classes in a discrete distribution.

    Notes
    -----
    Also known as categorical / multinomial / multinoulli in the literature.

    """

    def __init__(self, pvector, cvector):
        self.p = pvector
        self.c = cvector
        self.K = self.p.shape[0]
        self.I = len(self.c)

    def __repr__(self):
        i = "Discrete Multinomial Emissions\n"
        b = "" + "Emmission Probability: \n %s \n" % str(self.p)
        return i + b

    def sample(self, k):
        x = np.random.multinomial(1, self.p[k])
        return np.where(x == 1)[0][0]

    def fit(self, obs, logweights):
        logGamma = np.concatenate(logweights, 1)
        normalizer = np.exp(logGamma - logsumexp(logGamma, axis=1)[:, np.newaxis])
        for k in range(self.K):
            self.p[k] = np.exp(
                np.log(np.sum(normalizer[k, :] * obs, 1))
                - np.log(np.sum(normalizer[k, :]))
            )

    def loglikelihood(self, y, state):
        return np.log(self.p[state, y[0]])

    def likelihood(self, y, state):
        return np.exp(self.loglikelihood(y, state))
