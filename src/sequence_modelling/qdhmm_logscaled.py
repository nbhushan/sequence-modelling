# -*- coding: utf-8 -*-
"""
QDHMM in logscale to deal with numerical issues

@author: nbhushan

"""

import numpy as np
from sequence_modelling.utils import logsumexp
from scipy import sparse
import matplotlib.pyplot as plt
import logging, sys
import pdb
import time

# set up the logger
logger = logging.getLogger(__name__)
hdlr = logging.StreamHandler(sys.stderr)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

# np.seterr(all='ignore')


class HMMPlus:
    """
    HMM plus class to define params

    Params:
        p = Initial probability distribution for the active state
        zeta = probability of self transitions in active state
        eta = probability of self transitions in inactive state

    Atrributes:
        D : Number of States (excluding active and state with infinite timeout).
        N: length of observation sequence.

    """

    def __init__(self, p, zeta, eta, D, O):
        """
        Initialize the parameters of the HMM
        where A[K+1,K] is the state transition matrix
        and O is the emmision model object
        """
        self.D = D + 2
        self.logzeta = np.log(zeta)
        self.logeta = np.log(eta)
        self.O = O
        self.logp = np.log(p)
        self.logA = None

    def __repr__(self):
        p = "" + "p:\n %s" % (str(np.exp(self.logp)))
        zeta = "" + "\nzeta: \n %s" % (str(np.exp(self.logzeta)))
        eta = "" + "\neta: \n %s" % (str(np.exp(self.logeta)))
        return p + zeta + eta

    def sample(self, dim, N):
        """
        generates a sequence of length  N
        """
        np.random.seed(10)
        zeta = np.exp(self.logzeta)
        eta = np.exp(self.logeta)
        w = np.zeros(self.D)
        p = np.exp(self.logp)
        w[0] = p
        w[1:] = (1 - p) / (self.D - 1)
        obs = np.zeros((dim, N))
        zes = np.zeros((N))
        # Base case, when n=0, select initial state
        d = np.where(np.random.multinomial(1, w) == 1)[0][0]

        # generate obs using the hmm model
        for n in range(N):
            zes[n] = d
            obs[:, n] = self.O.Sample(d)
            if d == 0:
                d = int(np.random.multinomial(1, [zeta, (1 - zeta)])[0] == 0)
            else:
                d = (
                    min(d + 1, self.D - 1)
                    if np.random.multinomial(1, [(1 - eta), eta])[0] == 0
                    else 0
                )
        return obs, zes

    def viterbi(self, obs, tau):
        """
        Computes Most probable path based on Viterbi algorithm
        Most probable state sequence = argmax_zP(Z|X)
        """
        # pdb.set_trace()
        D = self.D
        N = obs.shape[1]
        logB = np.zeros((D, N))
        w = np.zeros((D))
        path = np.zeros((N), int)
        prob = np.zeros((D, D))
        V = np.zeros((D, N))
        p = np.exp(self.logp)
        w[0] = p
        w[1:] = (1 - p) / (self.D - 1)
        logw = np.log(w)
        # calcualte emission likelihoods
        logB = self.O.Loglikelihood(obs)
        self.logA = self.buildTransmat()
        # Base case when n = 0
        V[:, 0] = logw + logB[:, 0]

        # Track Maximal States
        psi = np.zeros((self.D, N), int)

        # Induction
        for n in range(1, N):
            for k in range(self.D):
                # pdb.set_trace()
                prob = V[:, n - 1][:, np.newaxis] + self.logA[:, k]
                V[k, n] = np.max(prob, 0) + logB[k, n]
                psi[k, n] = np.argmax(prob, axis=0)

        # calculate sequence through most lilely states
        path[-1] = np.argmax(V[:, -1][:, np.newaxis], 0)
        for n in range(N - 2, -1, -1):
            path[n] = psi[path[n + 1], n + 1]
        test = np.digitize(path, tau, right=True)
        return test

    def alpha(self, logB):
        """
        Compute alpha values
        where: alpha [K,N]
        and alpha [i,n] =  joint probability of being in state i
        after observing 1..N observations.
        """
        D = self.D
        N = logB.shape[1]
        assert logB.shape == (D, N)
        logAlpha = np.zeros((D, N), dtype=np.float)

        w = np.zeros((D))
        p = np.exp(self.logp)
        w[0] = p
        w[1:] = (1 - p) / (self.D - 1)
        logw = np.log(w)
        to = time.time()
        # Base case, when n=0
        logAlpha[:, 0] = logw + logB[:, 0]
        # pdb.set_trace()
        # Induction
        for n in range(1, N):
            logAlpha[:, n] = (
                logsumexp(self.logA[:, :].T + logAlpha[:, n - 1], 1) + logB[:, n]
            )
        logger.debug("Time to compute alpha matrix : %.5f s " % (time.time() - to))
        return logAlpha

    def beta(self, logB):
        """
        compute beta values
        where: beta[K,N]
        beta [i,n] =  probability generating observations X_n+1..X_N,
        given Z_n
        """
        D = self.D
        N = logB.shape[1]
        logBeta = np.zeros((D, N), dtype=np.float)
        to = time.time()
        # Base case when n = N
        logBeta[:, -1] = 0.0
        # pdb.set_trace()
        # Induction
        for n in range(N - 2, -1, -1):
            logBeta[:, n] = logsumexp(
                logBeta[:, n + 1] + self.logA[:, :] + logB[:, n + 1], axis=1
            )
        logger.debug("Time to compute beta matrix : %.5f s " % (time.time() - to))
        return logBeta

    def gammaKsi(self, logB):
        """
        computes gamma values
        where Gamma[K,N]
        gamma[i,n] = probability of being in state 'i' at time't' given the
        complete observation sequence 1..N
        gamma(Z_n) = alpha(Z_n).beta(Z_n) / P(X)
        and ksi[N,K,K]  = joint posteriot probability of two succesive hidden
        states.
        """
        # pdb.set_trace()
        D = self.D
        N = logB.shape[1]
        logGamma = np.zeros((D, N), dtype=np.float)
        temp_array = np.zeros((self.D, self.D))
        temp_zeta = np.zeros((N - 1))
        temp_eta = np.zeros((N - 1))

        logAlpha = self.alpha(logB)
        logBeta = self.beta(logB)

        loglikelihood = logsumexp(logAlpha[:, -1])
        # compute gamma and normalize
        to = time.time()
        logGamma = logAlpha + logBeta
        logGamma -= logsumexp(logGamma, 0)
        logger.debug("Time to compute gamma matrix : %.5f s " % (time.time() - to))

        # Estimate Ksi. Ksi is too large to fit in contiguous memory.
        # Estimate Ksi at every time step n, and store the relevane parameters
        # required for the computation and zeta and eta
        to = time.time()
        for n in range(N - 1):
            temp = logB[:, n + 1] + logBeta[:, n + 1]
            temp_array[:, :] = logAlpha[:, n][:, np.newaxis] + self.logA[:, :] + temp
            # Normalize
            temp_array -= logsumexp(temp_array.flatten())
            temp_zeta[n] = temp_array[0, 0]
            temp_eta[n] = logsumexp(temp_array[1:, 1:])
        logger.debug(
            "Time to compute (ksi) zera and eta arrays : %.5f s " % (time.time() - to)
        )
        return (loglikelihood / N, logGamma, logsumexp(temp_zeta), logsumexp(temp_eta))

    def hmmFit(self, obs, maxiter=50, epsilon=0.000001, visualize=False, debug=True):
        """
        Performs iterations of the EM (baum welch)
        algorithm until convergence
        """
        # pdb.set_trace()
        if debug:
            logger.setLevel(logging.DEBUG)
        logger.info("Running the EM algorithm..")

        numseq = len(obs)
        lastavgloglikelihood = -np.inf
        logGammalist = [np.newaxis] * numseq
        loglikelihoodlist = [np.newaxis] * numseq
        logzetalist = [np.newaxis] * numseq
        logetalist = [np.newaxis] * numseq
        ll = []

        for iteration in range(maxiter):
            start_time = time.time()
            logger.debug(" ------------------------------------------------")
            logger.debug("iter: %d" % iteration)

            # create the sparse transition matrix A
            to = time.time()
            self.logA = self.buildTransmat()
            logger.debug(
                "Time to create sparse transition matrix : %.5f s " % (time.time() - to)
            )

            # Estimate posteriors for individual sequnces
            for seq, obsseq in enumerate(obs):
                # E-step
                # calcualte the posterior probability for each sequence

                # compute the emission distribution matrix
                to = time.time()
                logB = self.O.Loglikelihood(obsseq)
                logger.debug(
                    "Time to create emission distribution matrix : %.5f s "
                    % (time.time() - to)
                )
                logger.debug("E step..")
                (
                    loglikelihoodlist[seq],
                    logGammalist[seq],
                    logzetalist[seq],
                    logetalist[seq],
                ) = self.gammaKsi(logB)

            meanloglikelihood = np.sum(loglikelihoodlist)
            ll.append(meanloglikelihood)
            logger.debug("LLH : %.18f" % meanloglikelihood)
            # convergence criteria
            if abs(meanloglikelihood - lastavgloglikelihood) < epsilon:
                logger.info("Convergence after %d iterations" % iteration)
                break
            lastavgloglikelihood = meanloglikelihood
            if iteration == maxiter - 1:
                logger.info("No convergence after %d iterations " % (iteration + 1))
                break

            # concatenate the observations sequences
            obsarray = np.concatenate(obs, 1)

            # M-step
            # Maximise the model parameters based on the posterior
            logGamma = np.concatenate(logGammalist, 1)
            logger.debug("M step..")
            # Re-estimate p
            self.logp = logGamma[0, 0]
            # Re-estimate zeta
            self.logzeta = logsumexp(logzetalist) - logsumexp(logGamma[0, :-1])
            logger.debug("zeta: %0.8f" % np.exp(self.logzeta))
            # Re-estiamte eta
            self.logeta = logsumexp(logetalist) - logsumexp(logGamma[1:, :-1])
            logger.debug("eta: %0.8f" % np.exp(self.logeta))
            # Re-estimate the emission model parameters
            to = time.time()
            self.O.Fit(obsarray, logGamma)
            logger.debug("Time to update emission model : %.5f s " % (time.time() - to))
            end_time = time.time()
            logger.debug("Time to run iteration : %.5f s " % (end_time - start_time))

        return (meanloglikelihood * numseq, ll)

    def buildTransmat(self):
        A = sparse.lil_matrix((self.D, self.D))
        diag = np.zeros(self.D) + np.exp(self.logeta)
        A.setdiag(diag, k=1)
        A[0, 0] = np.exp(self.logzeta)
        A[0, 1] = 1 - A[0, 0]
        A[-1, -1] = np.exp(self.logeta)
        A[1:, 0] = 1 - np.exp(self.logeta)
        return np.log(A.todense())

    # TODO: Isolate visualization code
    def formatdata(self, obs, path, mean):
        y = obs.flatten()
        z = path
        m = mean.flatten()
        ordr = np.argsort(m)
        return y, np.argsort(ordr)[z], m[ordr]

    def displayviterbi(self, ax, y, z, m):
        # pdb.set_trace()
        r = np.arange(len(y))
        ax.plot(r, y)
        for k, v in enumerate(m):
            t = z == k
            ax.plot(
                r[t],
                v * np.ones((t.sum(),), float),
                ".",
                label="State {0}: {1:g}".format(k, v),
            )
        ax.set_title("HMM Viterbi State sequence prediction")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Power (W)")
        ax.legend(prop=dict(size="xx-small"))

    def plotconvergence(self, ax, ll):
        x = list(range(len(ll)))
        ax.plot(x, ll)
        ax.set_title("EM convergence")
        ax.set_ylabel("LLH")
        ax.set_xlabel("Iter")
