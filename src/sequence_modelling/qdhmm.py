# -*- coding: utf-8 -*-
"""
The QDHMM object

@author: nbhushan

"""

import numpy as np
from scipy import sparse
import logging, sys
import time

# set up the logger
logger = logging.getLogger(__name__)
hdlr = logging.StreamHandler(sys.stderr)
logger.addHandler(hdlr)
logger.setLevel(logging.ERROR)


class QDHMM:
    """The QDHMM object.

    Attributes
    ----------
    p : float
            initial probability distribution for the active state.
    zeta : float
            probability of self transitions in active state.
    eta : float
            probability of self transitions in inactive state.
    O : object
            QDHMM Emission Model



    Notes
    -----
    The QDHMM is an extension to a standard HMM.

    """

    def __init__(self, p, zeta, eta, O):
        """
        Initialize the parameters of the HMM
        where A[K+1,K] is the state transition matrix
        and O is the emmision model object
        """
        self.zeta = zeta
        self.eta = eta
        self.O = O
        self.D = self.O.D
        self.p = p
        self.A = None

    def __repr__(self):
        p = "" + "p:\n %s" % (str(self.p))
        zeta = "" + "\nzeta: \n %s" % (str(self.zeta))
        eta = "" + "\neta: \n %s" % (str(self.eta))
        o = "" + "\nObservation distribution: \n %s" % self.O
        return p + zeta + eta + o

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

        w = np.zeros(self.D)
        p = self.p
        w[0] = p
        w[1:] = (1 - p) / (self.D - 1)
        obs = np.zeros((dim, N))
        zes = np.zeros((N))
        # Base case, when n=0, select initial state
        d = np.where(np.random.multinomial(1, w) == 1)[0][0]
        # generate obs using the hmm model
        for n in range(N):
            zes[n] = d
            # obs[:,n] = self.O.Sample(d)
            if d == 0:
                d = int(np.random.multinomial(1, [self.zeta, (1 - self.zeta)])[0] == 0)
            else:
                d = (
                    min(d + 1, self.D - 1)
                    if np.random.multinomial(1, [(1 - self.eta), self.eta])[0] == 0
                    else 0
                )
        obs[:, :] = self.O.Sample(zes)
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
        # obs=obs[0]
        tau = self.O.tau
        to = time.time()
        # pdb.set_trace()
        D = self.D
        N = obs.shape[1]
        B = np.zeros((D, N))
        w = np.zeros((D))
        path = np.zeros((N), int)
        # prob=np.zeros((D,D))
        V = np.zeros((D, N), float)
        c = np.ones((N))
        p = self.p
        w[0] = p
        w[1:] = (1 - p) / (self.D - 1)
        # calcualte emission likelihoods
        B = self.O.likelihood(obs)
        self.A = self.buildTransmat()
        self.A = np.array(self.A.todense())
        # Base case when n = 0
        V[:, 0] = w * B[:, 0]
        if np.max(V[:, 0]) != 0.0:
            c[0] = 1.0 / np.max(V[:, 0])
            V[:, 0] *= c[0]
        # Track Maximal States
        psi = np.zeros((self.D, N), int)
        # pdb.set_trace()
        # Induction
        for n in range(1, N):
            for k in range(D):
                prob = V[:, n - 1] * self.A[:, k]
                # prob=self.A.getcol(k).multiply(V[:,n-1:n])
                V[k, n] = np.max(prob, 0) * B[k, n]
                psi[k, n] = np.argmax(prob, axis=0)
            if np.max(V[:, n]) != 0.0:
                c[n] = 1.0 / np.max(V[:, n])
                V[:, n] *= c[n]
        # calculate sequence through most lilely states
        path[-1] = np.argmax(V[:, -1][:, np.newaxis], 0)
        for n in range(N - 2, -1, -1):
            path[n] = psi[path[n + 1], n + 1]
        test = np.digitize(path, tau, right=True)
        logger.debug("Time to compute viterbi path : %.5f s " % (time.time() - to))
        return test

    def alpha(self, B):
        """Compute alpha (forward) values.

            alpha [i,n] =  joint probability of being in state i,
            after observing 1..N observations.   .

        Parameters
        ----------
        B : ndarray
            The observation probability matrix.

        Returns
        -------
        alphahat : ndarray
            The scaled alpha values.
        c : ndarray
            The scaling factors.

        Notes
        -----
        Refer to Rabiner's paper [1]_ for the scaling factors used here.


        References
        ----------
        .. [1] Rabiner, L. A tutorial on hidden Markov models and selected
            applications in speech recognition Proceedings of the IEEE,
            1989, 77, 257-286.

        """
        # pdb.set_trace()
        D = self.D
        N = B.shape[1]
        assert B.shape == (D, N)
        Alpha = np.zeros((D, 1), dtype=np.float64)
        Alphahat = np.zeros((D, N), dtype=np.float64)
        c = np.ones((N))
        w = np.zeros((D))
        p = self.p
        w[0] = p
        w[1:] = (1 - p) / (self.D - 1)
        to = time.time()
        # Base case, when n=0
        Alphahat[:, 0] = Alpha[:, 0] = w * B[:, 0]
        if np.sum(Alpha[:, 0]) != 0:
            c[0] = 1.0 / np.sum(Alpha[:, 0])
            Alphahat[:, 0] = c[0] * Alpha[:, 0]
        # Induction
        for n in range(1, N):
            Alphahat[:, n] = (self.A.T.dot(Alphahat[:, n - 1])) * B[:, n]
            if np.sum(Alphahat[:, n]) != 0:
                c[n] = 1.0 / np.sum(Alphahat[:, n])
                Alphahat[:, n] = c[n] * Alphahat[:, n]
        logger.debug("Time to compute alpha matrix : %.5f s " % (time.time() - to))
        return Alphahat, c

    def beta(self, B, c):
        """Compute beta (backward) values.

            beta [i,n] =  conditional probability generating observations
            Y_n+1..Y_N, given Z_n.

        Parameters
        ----------
        B : ndarray
            The observation probability matrix.
        c : ndarray
            The scaling factors obtained from the alpha computation

        Returns
        -------
        betahat : ndarray
            The scaled beta values.

        Notes
        -----
        DO NOT call the beta function before calling the alpha function.
        Refer to Rabiner's paper [1]_ for the scaling factors used here.

        References
        ----------
        .. [1] Rabiner, L. A tutorial on hidden Markov models and selected
            applications in speech recognition Proceedings of the IEEE,
            1989, 77, 257-286.

        """

        D = self.D
        N = B.shape[1]
        Betahat = np.zeros((D, N), dtype=float)
        to = time.time()
        # Base case when n = N
        Betahat[:, -1] = 1.0
        Betahat[:, -1] = c[-1] * Betahat[:, -1]
        # Induction
        for n in range(N - 2, -1, -1):
            Betahat[:, n] = self.A.dot((c[n] * Betahat[:, n + 1] * B[:, n + 1]))
        logger.debug("Time to compute beta matrix : %.5f s " % (time.time() - to))
        return Betahat

    def gammaKsi(self, B):
        """Compute gamma (posterior distribution) values.

            gamma [i,n] =  conditional probability of the event state 'i'
            at time 'n', given the complete observation sequence.

        Parameters
        ----------
        B : ndarray
            The observation probability matrix.


        Returns
        -------
        llh : float
            The normalized log-likelihood.
        gamma : ndarray
            The posterior distribution.
        float
            The number of tranisitions into the active state.
        float
            The number of transitions into the inactive states.

        Notes
        -----
        Ksi is the joint succesive posterior distrbution.
        ksi[n,i,j]  = joint posterior probability of two succesive hidden
        states 'i' and 'j' at time 'n'.
        In the QDHMM, Ksi is too large to fit in contiguous memory [N,K,K].
        Hence we estimate Ksi at every time step n, and store the relevant
        parameters required for the computation and zeta and eta
        (the transition parameters)

        """
        N = B.shape[1]
        _temp_zeta = np.zeros((N - 1))
        _temp_eta = np.zeros((N - 1))
        Alpha, c = self.alpha(B)
        Beta = self.beta(B, c)
        loglikelihood = -np.sum(np.log(c))
        # compute gamma and normalize
        to = time.time()
        Gamma = Alpha * Beta / c
        Gamma /= np.sum(Gamma, 0)
        logger.debug("Time to compute gamma matrix : %.5f s " % (time.time() - to))
        to = time.time()
        for n in range(N - 1):
            temp_array = sparse.csc_matrix(
                self.A.multiply(Alpha[:, n : n + 1] * B[:, n + 1] * Beta[:, n + 1]),
                dtype=float,
            )
            # Normalize
            temp_array.data /= np.sum(temp_array.data)
            _temp_zeta[n] = temp_array[0, 0]
            _temp_eta[n] = 1.0 - (
                temp_array[0, 1]
                + np.sum(temp_array.data[temp_array.indptr[0] : temp_array.indptr[1]])
            )
        logger.debug(
            "Time to compute (ksi) zera and eta arrays : %.5f s " % (time.time() - to)
        )
        return (loglikelihood / N, Gamma, np.sum(_temp_zeta), np.sum(_temp_eta))

    def fit(self, obs, maxiter=50, epsilon=1e-5, debug=False, metaheuristic="local"):
        """Fit the QDHMM to the given data using the (adapted Baum-Welch)
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
        metaheuristic : {'local', 'sa', 'genetic'}, optional
            The meta-heuristic to be used to solve the objective in the M-step.
            'local' is simple local search. 'genetic' is genetic algorithm and
            'sa' is simulated annealing.


        Returns
        -------
        float
            The normalized log-likelihood.
        list
            The list of log-likelihoods for each iteration of the EM algorithm.
            To check for monotonicity of the log-likelihoods.

        """
        # pdb.set_trace()
        if debug:
            logger.setLevel(logging.DEBUG)
        logger.info("Running the QDHMM EM algorithm..")
        numseq = len(obs)
        lastavgloglikelihood = -np.inf
        Gammalist = [np.newaxis] * numseq
        loglikelihoodlist = [np.newaxis] * numseq
        zetalist = [np.newaxis] * numseq
        etalist = [np.newaxis] * numseq
        ll = []
        estimatetau = True
        for iteration in range(maxiter):
            start_time = time.time()
            logger.debug(" ------------------------------------------------")
            logger.debug("iter: %d" % iteration)
            # create the sparse transition matrix A
            self.A = self.buildTransmat()
            # Estimate posteriors for individual sequnces
            for seq, obsseq in enumerate(obs):
                # E-step
                B = self.O.likelihood(obsseq)
                logger.debug("E step..")
                (
                    loglikelihoodlist[seq],
                    Gammalist[seq],
                    zetalist[seq],
                    etalist[seq],
                ) = self.gammaKsi(B)

            meanloglikelihood = np.mean(loglikelihoodlist)
            ll.append(meanloglikelihood)
            logger.debug("LLH : %.18f" % meanloglikelihood)
            # convergence criteria
            if abs(meanloglikelihood - lastavgloglikelihood) < epsilon:
                logger.info("Convergence after %d iterations" % iteration)
                break
            lastavgloglikelihood = meanloglikelihood
            # concatenate the observations sequences
            obsarray = np.concatenate(obs, 1)

            # M-step
            Gamma = np.concatenate(Gammalist, 1)
            logger.debug("M step..")
            self.p = Gamma[0, 0]
            self.zeta = np.sum(zetalist) / np.sum(Gamma[0, :-1])
            self.eta = np.sum(etalist) / np.sum(Gamma[1:, :-1])
            self.O.Fit(obsarray, Gamma, estimatetau, metaheuristic)
            estimatetau = (estimatetau + 1) % 2
            end_time = time.time()
            logger.debug("Time to run iteration : %.5f s " % (end_time - start_time))
            if iteration == maxiter - 1:
                logger.info("No convergence after %d iterations " % (iteration + 1))
                break
        return (meanloglikelihood, ll)

    def buildTransmat(self):
        """Builds the sparse transition matrix.

        Returns
        -------
        A : scipy.sparse.csr_matrix
            The sparse transition matrix.

        """
        A = sparse.lil_matrix((self.D, self.D), dtype=float)
        diag = np.zeros(self.D) + self.eta
        A.setdiag(diag, k=1)
        A[0, 0] = self.zeta
        A[0, 1] = 1 - A[0, 0]
        A[-1, -1] = self.eta
        A[1:, 0] = 1 - self.eta
        return sparse.csr_matrix(A)
