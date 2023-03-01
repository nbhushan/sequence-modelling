# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:05:45 2013

@author: nbhushan
"""

import os, sys

dirname = os.path.dirname
sys.path.append(os.path.join(dirname(dirname(__file__))))

import numpy as np
import sequence_modelling.emmissions as emissions
from sequence_modelling.hmm import StandardHMM
import sequence_modelling.emissionplus as emissionplus
from sequence_modelling.qdhmm import QDHMM
import sequence_modelling.hmmviz as viz


def test():
    """
    This function has the following workflow:
        1) Define an initial gaussian emmission object
        2) Generate observations from the emmission object
        3) Define intial A , where pi = A[-1]
        4) Train the model and find params which best fit the observations
        5) Visualize the state sequence.
    """
    # adjust the precision of printing float values
    np.set_printoptions(precision=4, suppress=True)

    # Initialize model parameters
    tau = np.array([0, 8, 14])
    p, zeta, eta, D = 0.001, 0.45, 0.97, np.max(tau)

    # Create an  QDHMM and Gaussian emmission object to sample data
    emmissionModel = emissionplus.Gaussian(
        mu=np.array([[532.23915, 86.69044, 45.2, 26.552]]),
        var=np.array([[[7568.806]], [[58.944]], [[28.944]], [[2.025350]]]),
        tau=tau,
    )
    samplemodel = QDHMM(p, zeta, eta, emmissionModel)

    # sample data from he emmission model

    N = [1000, 1000, 600, 1000]
    dim = 1
    numseq = 4
    obs = [np.newaxis] * numseq
    for n in range(numseq):
        obs[n], zes = samplemodel.sample(dim, N[n])

    for seq in range(len(obs)):
        print(
            "Length of observation sequence " + str(seq) + " :" + str(obs[seq].shape[1])
        )

        # Initialize transition Matrix
    A = np.array(
        [
            [0.3832979, 0.6167021, 0.0, 0.0],
            [0.3507986, 0.3821364, 0.2670650, 0.0],
            [0.3494203, 0.0, 0.3399672, 0.3106124],
            [0.2237589, 0.0, 0.0, 0.7762411],
            [1.0 / 4, 1.0 / 4, 1.0 / 4, 1.0 / 4],
        ]
    )

    # Create an  HMM and Gaussian emmission object
    emmissionModel = emissions.Gaussian(
        mu=np.array([[932.23915, 46.69044, 45.2, 26.552]]),
        covar=np.array([[[7568.806]], [[158.944]], [[78.944]], [[8.025350]]]),
    )
    trainmodel = StandardHMM(A, emmissionModel)

    print("\n Initial HMM model:\n ")
    print(trainmodel)
    print()
    print("\nInitial Emmission model: \n", trainmodel.O)
    print("*" * 80)

    # Fit the model to the data and print results
    newloglikehood, ll, duration, rankn, res = trainmodel.hmmFit(
        obs, maxiter=20, epsilon=1e-6, debug=True
    )
    path = [None] * numseq
    for seq in range(numseq):
        path[seq] = trainmodel.viterbi(obs[seq])

    print("Log likelihood: \n", newloglikehood)
    print("Re-estimated HMM Model: \n", trainmodel)
    print("Re-estimated Emmission model: \n", trainmodel.O)

    # Viterbi state duration
    lengths = [None] * numseq
    for idx, stateseq in enumerate(path):
        lengths[idx] = trainmodel.estimateviterbiduration(stateseq)
    for length in lengths:
        print(
            "Viterbi state duration:",
            np.max(length[0]),
            np.max(length[1]),
            np.max(length[2]),
            np.max(length[3]),
        )
    print("Posterior distribution duration estimation:", duration)

    # Visualize
    uniqueid = "output/qdhmmtest"
    from matplotlib.pyplot import figure, show
    from matplotlib.backends.backend_pdf import PdfPages

    pp = PdfPages(uniqueid + ".pdf")
    for seq in range(numseq):
        fa = figure()
        viz.view_viterbi(fa.add_subplot(1, 1, 1), obs, path, trainmodel.O.mu, seq)
        fa.tight_layout()
        pp.savefig()
        fb = figure()
        viz.view_postduration(
            fb.add_subplot(111), obs, path, trainmodel.O.mu, res, rankn, seq
        )
        fb.tight_layout()
        pp.savefig()
    fc = figure()
    viz.view_EMconvergence(fc.add_subplot(1, 1, 1), ll)
    pp.savefig()
    pp.close()


if __name__ == "__main__":
    test()
