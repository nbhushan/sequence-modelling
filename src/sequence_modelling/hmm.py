# -*- coding: utf-8 -*-
"""
The HMM class

@author: nbhushan

"""

import logging, sys, time, pdb
import numpy as np
from sequence_modelling.utils import logsumexp

# set up the logger
logger = logging.getLogger(__name__)
hdlr = logging.StreamHandler(sys.stderr)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)


class StandardHMM:
    """The standard HMM object.

    Attributes
    ----------
    A : ndarray
        The transition distribution.
    O : object
        The HMM emission model.

    Notes
    -----
    pi, the initial state distribution is the last row of the transition
    matrix 'A'. i.e.  pi = A[-1].
    Number of States: K = A.shape[1]

    Examples
    --------
    >>> # import the only external dependency
    >>> import numpy as np
    >>> #import package modules
    >>> from sequence_modelling.hmm import StandardHMM
    >>> from sequence_modelling.emissions import Gaussian
    >>> from sequence_modelling import hmmviz
    ...
    >>> # Define the model parameters
    >>> # the transition matrix A
    >>> A = np.array([[0.9, 0.1, 0.0],
    ...               [0.0, 0.9, 0.1],
    ...               [0.0, 0.0, 1.0]])
    >>> # the emission object B
    >>> B = Gaussian(mu = np.array([[0.0, 1.0, 2.0],
    ...                             [0.0, 1.0, 2.0]]),
    ...              covar = np.array([[0.1, 0.1, 0.1],
    ...                              [0.1, 0.1, 0.1]]))
    ...
    >>> # Build the HMM model object
    >>> hmm = StandardHMM(A, B)
    """

    def __init__(self, A, O):
        """
        Initialize the parameters of the HMM
        where A[K+1,K] is the state transition matrix
        and O is the emmision model object
        """
        self.K = A.shape[1]
        assert A.shape[0] == self.K + 1
        self.O = O
        self.logA = np.log(A)

    def __repr__(self):
        return (
            f"nStates: {self.K}\n"
            f"A:\n {self.logA[:-1].round(2)}\n"
            f"Pi: \n {self.logA[-1].round(2)}\n"
            f"Observation distribution: \n {self.O}"
        )

    def sample(self, dim=1, N=1000):
        """Generates an observation sequence of length N.

        Parameters
        ----------
        dim : int
            The dimension of the data (univariate=1, etc..).
        N : int
            The length of the observation sequence.

        Returns
        -------
        obs : ndarray
            An array of N observations.
        zes : ndarray
            The state sequence that generated the data.

        """
        obs = np.zeros((dim, N))
        zes = np.zeros((N), int)
        # Base case, when n=0, select initial state
        k = np.where(np.random.multinomial(1, np.exp(self.logA[-1])) == 1)[0][0]
        # generate obs using the hmm model
        for n in range(N):
            zes[n] = k
            # obs[:,n] =   self.O.Sample(k)
            k = np.where(np.random.multinomial(1, np.exp(self.logA[k])) == 1)[0][0]
        obs[:, :] = self.O.sample(zes)
        return obs, zes

    def viterbi(self, obs):
        """Computes Most probable path based on Viterbi algorithm.

           Most probable state sequence = argmax_z P(Z|X)

        Parameters
        ----------
        obs : array_like
            Observation sequence.

        Returns
        -------
        path : ndarray
            The Viterbi decoded state sequence.

        Notes
        -----
        Refer to Rabiner's paper [1]_ or the original Viterbi paper [2]_.


        References
        ----------
        .. [1] Rabiner, L. A tutorial on hidden Markov models and selected
            applications in speech recognition Proceedings of the IEEE,
            1989, 77, 257-286.
        .. [2] Viterbi, A. Error bounds for convolutional codes and an
            asymptotically optimum decoding algorithm Information Theory,
            IEEE Transactions on, 1967, 13, 260-269

        """
        K = self.K
        N = obs.shape[1]
        logB = np.zeros((K, N))
        path = np.zeros((N), int)
        # calculate emission likelihoods
        logB = self.O.loglikelihood(obs)
        assert logB.shape == (K, N)
        V = np.zeros((K, N))
        # Base case when n = 0
        V[:, 0] = self.logA[-1] + logB[:, 0]
        # Track Maximal States
        psi = np.zeros((self.K, N), int)
        # Induction
        for n in range(1, N):
            for k in range(self.K):
                prob = V[:, n - 1] + self.logA[:-1][:, k]
                V[k, n] = np.max(prob, 0) + logB[k, n]
                psi[k, n] = np.argmax(prob, axis=0)
        # calculate sequence through most lilely states
        path[-1] = np.argmax(V[:, -1])
        for n in range(N - 2, -1, -1):
            path[n] = psi[path[n + 1], n + 1]
        return path

    def alpha(self, logB):
        """Compute alpha (forward) distribution.

            alpha [i,n] =  joint probability of being in state i,
            after observing 1..N observations.   .

        Parameters
        ----------
        logB : ndarray
            The observation probability matrix in logarithmic space.

        Returns
        -------
        logalpha : ndarray
            The log scaled alpha distribution.

        Notes
        -----
        Refer to Tobias Man's paper [1]_ for the motivation behind the
        scaling factors used here. Note that this scaling methods is suitable
        when the dynamics of the system is not highly sparse. Adaptation of
        log-scaling in the QDHMM would require the use to construct a new
        sparse data structure

        References
        ----------
        .. [1] Mann, T. P. Numerically Stable Hidden Markov Model
               Implementation 2006.

        """
        K = self.K
        N = logB.shape[1]
        assert logB.shape == (K, N)
        logAlpha = np.zeros((K, N), dtype=np.float64)
        # Base case, when n=0
        logAlpha[:, 0] = self.logA[-1] + logB[:, 0]
        # induction
        for n in range(1, N):
            logAlpha[:, n] = (
                logsumexp(self.logA[:-1][:, :].T + logAlpha[:, n - 1], 1) + logB[:, n]
            )
        return logAlpha

    def beta(self, logB):
        """Compute beta (backward) distribution.

            beta [i,n] =  conditional probability generating observations
            Y_n+1..Y_N, given Z_n.

        Parameters
        ----------
        logB : ndarray
            The observation probability matrix in logarithmic space.

        Returns
        -------
        logbeta : ndarray
            The log scaled beta distribution.

        Notes
        -----
        Refer to Tobias Man's paper [1]_ for the motivation behind the
        scaling factors used here. Note that this scaling methods is suitable
        when the dynamics of the system is not highly sparse. Adaptation of
        log-scaling in the QDHMM would require the use to construct a new
        sparse data structure.

        References
        ----------
        .. [1] Mann, T. P. Numerically Stable Hidden Markov Model
               Implementation 2006.

        """
        K = self.K
        N = logB.shape[1]
        logBeta = np.zeros((K, N), dtype=np.float64)
        # Base case when n = N
        logBeta[:, -1] = 0.0
        # Induction
        for n in range(N - 2, -1, -1):
            logBeta[:, n] = logsumexp(
                logBeta[:, n + 1] + self.logA[:-1][:, :] + logB[:, n + 1], 1
            )
        return logBeta

    def gammaKsi(self, logB):
        """Compute gamma (posterior distribution) and Ksi (joint succesive
            posterior distrbution) values.

        gamma [i,n] =  conditional probability of the event state 'i'
        at time 'n', given the complete observation sequence.

        ksi[n,i,j]  = joint posterior probability of two succesive hidden
        states 'i' and 'j' at time 'n'.


        Parameters
        ----------
        logB : ndarray
            The observation probability matrix in logarithmic space.


        Returns
        -------
        llh : float
            The normalized log-likelihood.
        logGamma : ndarray
            The log posterior distribution.
        logKsi : ndarray
            The log joint posterior probability distribution.
        logAlpha : ndarray
            The log scaled alpha distribution.
        logBeta : ndarray
            The log scaled beta distribution.


        """

        K = self.K
        N = logB.shape[1]
        logKsi = np.zeros((N - 1, K, K), dtype=np.float64)
        logGamma = np.zeros((K, N), dtype=np.float64)
        logAlpha = self.alpha(logB)
        logBeta = self.beta(logB)
        loglikelihood = logsumexp(logAlpha[:, -1])
        # compute gamma
        logGamma = logAlpha + logBeta
        logGamma -= logsumexp(logGamma, 0)
        # compute ksi
        for n in range(N - 1):
            temp = logB[:, n + 1] + logBeta[:, n + 1]
            logKsi[n, :, :] = (
                logAlpha[:, n][:, np.newaxis] + self.logA[:-1][:, :] + temp
            )
            logKsi[n, :, :] -= logsumexp(logKsi[n, :, :].flatten())
        return (loglikelihood, logGamma, logKsi, logAlpha, logBeta)

    def fit(self, obs, maxiter=50, epsilon=0.0001, debug=True):
        """Fit the standard HMM to the given data using the (adapted Baum-Welch)
           EM algorithm.

        Parameters
        ----------
        obs : list
            The list of observations sequences where every sequence is a
            ndarray. The sequences can be of different length, but
            the dimension of the features needs to be identical.
        maxiter : int, optional
            The maximum number of iterations of the EM algorithm. Default = 50.
        epsilon : float, optional
            The minimum allowed threshold in the variation of the log-likelihood
            between succesive iterations of the EM algorithm. Once the variation
            exceeds 'epsilon' the algorithm is said to have converged.
            Default = 1e-6.
        debug : bool, optional
            Display verbose On/off.


        Returns
        -------
        float
            The normalized log-likelihood.
        list
            The list of log-likelihoods for each iteration of the EM algorithm.
            To check for monotonicity of the log-likelihoods.
        int
            The duration estimates of each HMM state from the posterior
            distribution.
        ndarray
            The top ranked 'n'  which are used to estimate the state
            durations.
        ndarray
            The expected value of the state durations obtained at
            the top ranked 'n'.

        """
        if debug:
            logger.setLevel(logging.DEBUG)
        logger.debug("Running the HMM EM algorithm..")
        ll = []
        numseq = len(obs)
        lastavgloglikelihood = -np.inf
        logksilist = [None] * numseq
        logGammalist = [None] * numseq
        logAlphalist = [None] * numseq
        logBetalist = [None] * numseq
        llhlist = [None] * numseq
        obsmatrix = [None] * numseq
        logB = [None] * numseq
        duration = [None] * numseq
        res = [None] * numseq
        rankn = [None] * numseq
        N = [None] * numseq

        for iteration in range(maxiter):
            start_time = time.time()
            logger.debug("-------------------------------------------")
            logger.debug("iter: %d" % iteration)
            logger.debug("E step..")
            for seq, obsseq in enumerate(obs):
                N[seq] = obsseq.shape[1]
                if self.O.__class__.__name__ == "Discrete":
                    obsmatrix[seq] = np.zeros((len(self.O.c), N))
                    obsmatrix[seq][np.int_(obsseq), list(range(N))] = 1
                # E-step
                # calcualte the posterior probability for each sequence
                logB[seq] = self.O.loglikelihood(obsseq)
                (
                    llhlist[seq],
                    logGammalist[seq],
                    logksilist[seq],
                    logAlphalist[seq],
                    logBetalist[seq],
                ) = self.gammaKsi(logB[seq])
            normalizellhlist = np.divide(llhlist, N)
            loglikelihood = np.sum(normalizellhlist)
            ll.append(loglikelihood)
            logger.debug("LLH: %0.10f" % (loglikelihood))
            if abs(loglikelihood - lastavgloglikelihood) < epsilon:
                for seq in range(len(obs)):
                    rankn[seq] = self.rankn(logksilist[seq])
                    duration[seq], res[seq] = self.estimatepostduration(
                        logAlphalist[seq],
                        logBetalist[seq],
                        logB[seq],
                        rankn[seq],
                        logGammalist[seq],
                        llhlist[seq],
                    )
                logger.info("Convergence after %d iterations" % iteration)
                break
            lastavgloglikelihood = loglikelihood
            obsarray = np.concatenate(obs, 1)
            if self.O.__class__.__name__ == "Discrete":
                obsarray = np.concatenate(obsmatrix, 1)
            # M-step
            logger.debug("M step..")
            self.logA[-1] = logsumexp(
                np.array([g[:, 0] for g in logGammalist]), axis=0
            ) - np.log(np.double(numseq))
            logKsiarray = np.concatenate(logksilist, axis=0)
            logGammasArray = np.concatenate([x[:, :-1] for x in logGammalist], axis=1)
            self.logA[:-1] = (
                logsumexp(logKsiarray, axis=0)
                - logsumexp(logGammasArray, axis=1)[:, np.newaxis]
            )
            self.O.fit(obsarray, logGammalist)
            end_time = time.time()
            logger.debug("Time to run iter : %.5f s" % (end_time - start_time))
            if iteration == maxiter - 1:
                for seq in range(len(obs)):
                    rankn[seq] = self.rankn(logksilist[seq])
                    duration[seq], res[seq] = self.estimatepostduration(
                        logAlphalist[seq],
                        logBetalist[seq],
                        logB[seq],
                        rankn[seq],
                        logGammalist[seq],
                        llhlist[seq],
                    )
                logger.info("No convergence after %d iterations" % (iteration + 1))
                break
        return (loglikelihood, ll, duration, rankn, res)

    def estimatepostduration(self, logalpha, logbeta, logB, rankn, g, llh):
        """Estimate state durations based on the posterior distribution.

        Since the durations are truncated by the timeout parameter, we use
        a distribution free method.

        Parameters
        -----------
        logalpha : ndarray
            Log scaled alpha distribution.
        logbeta : ndarray
            Log scaled beta values.
        logB : ndarray
            Observation probability distribution in log-space.
        rankn : ndarray
            the top ranked 'n' for eah state 'k', used to estimate state durations.
        g : ndarray
            log scaled posterior distribution ('logGamma')
        llh : float
            the normalized log-likelihood.

        Returns
        --------
        int
            The estimated durations in each state.
        ndarray
            The expected value of the state duration at the 'rankn'.

        Notes
        ------
        The QDHMM EM algorithm requires good initial estimates of the model
        parameters in order to converge to a good solution. We propose a
        distribution free method to find the expected value of state durations
        in a standard HMM model, which is then used to initialize the QDHMM
        'tau' parameters.

        """
        sub = len(rankn[0])
        N = logalpha.shape[1]
        K = self.K
        durations = np.zeros((K))
        res = np.zeros((K, sub))
        for o, k in enumerate(range(K)):
            inotk = set(range(K)) - set([k])
            for idx, n in enumerate(rankn[o]):
                const = np.zeros(len(inotk))
                # Base Case
                tmp = (N - 1 - n) * self.logA[k, k] - np.log(
                    1 - np.exp(self.logA[k, k])
                )
                # Induction
                for t in range(N - 2, n, -1):
                    tmp = np.logaddexp(
                        logbeta[k, t], logB[k, t + 1] + self.logA[k, k] + tmp
                    )
                for x, i in enumerate(inotk):
                    const[x] = (
                        logalpha[i, n]
                        + self.logA[i, k]
                        + logB[k, n + 1]
                        - (g[i, n] + llh)
                    )
                res[o, idx] = logsumexp(const) + tmp
        durations = np.max(np.exp(res), 1)
        return durations.astype(int), np.exp(res)

    def rankn(self, ksi, rank=10):
        """Find the top ranked 'n's used to estimate state durations.

        Find the top ranked 'n' s for which the posterior probability of
        transitioning into the state 'k' given we were not at state 'k'
        at time 'n-1'.

        Parameters
        -----------
        ksi : ndarray
            The joint sucesive posterior distribution in log-space
        rank : int, optional
            The number of the ranked 'n' which we chose to use to
            estimate state durations.

        Returns
        -------
        rankn : ndarray
            the top ranked 'n's for each state. Used to estimate state durations
        """
        # pdb.set_trace()
        ksi = np.exp(ksi)
        K = ksi.shape[1]
        rankn = [None] * K
        for k in range(K):
            rankn[k] = np.argsort(ksi[:, k - 1, k])[-rank:]
        return rankn

    def estimateviterbiduration(self, path):
        """Estimate the state durations based on the Viterbi decoded
            state sequence.

        Parameters
        -----------
        path : ndarray
            The Viterbi decoded state sequence.

        Returns
        --------
        int
            Estimated state durations based on the Viterbi path.

        """
        from itertools import groupby

        K = self.logA.shape[1]
        durations = [None] * K
        for k in range(K):
            a = path == k
            durations[k] = [sum(g) for b, g in groupby(a) if b]
        return durations
