# pylint: disable=import-error
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:31:26 2013

@author: nbhushan
"""

import time
import numpy as np
from sequence_modelling.emissionplus import Gaussian
from sequence_modelling.qdhmm import QDHMM
import sequence_modelling.hmmviz as viz


def test(heuristic):
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
    tau = np.array([0, 6, 9])
    p, zeta, eta = 0.00004, 0.75, 0.993

    # Create an  HMM and Gaussian emmission object to sample data
    emmissionModel = Gaussian(
        mu=np.array([[532.23915, 86.69044, 45.2, 26.552]]),
        var=np.array([[[7568.806]], [[58.944]], [[28.944]], [[2.025350]]]),
        tau=tau,
    )
    samplemodel = QDHMM(p, zeta, eta, emmissionModel)

    # sample data from he emmission model
    N = [10000]
    dim = 1
    numseq = 1
    obs = [np.newaxis] * numseq
    for n in range(numseq):
        obs[n], zes = samplemodel.sample(dim, N[n])
    print("number of sequences: ", len(obs))
    for seq in range(len(obs)):
        print("Length of observation sequence " + str(seq) + " :", obs[seq].shape[1])

    # create  EM initialization model
    _tau = np.array([0, 5, 7])

    _p, _zeta, _eta = 0.001, 0.3, 0.95
    _emmissionModel = Gaussian(
        mu=np.array([[702.77, 185.0, 66.90, 34.80]]),
        var=np.array([[[2000.0]], [[526.0]], [[140.0]], [[3.0]]]),
        tau=_tau,
    )

    initmodel = QDHMM(_p, _zeta, _eta, _emmissionModel)

    # state0data= obs[0][:,zes==0]
    print("Initial Model used to generate data\n", samplemodel)
    print(
        "Initial Model used to initialize EM obtained by a random guess \n", initmodel
    )
    start_time = time.time()
    likelihood, ll = initmodel.qdhmmFit(obs, 10, 1e-6, True, heuristic)
    end_time = time.time()
    print("Time taken to estimate parameters (s) :", (end_time - start_time))
    path = [None] * len(obs)
    for seq in range(len(obs)):
        path[seq] = initmodel.viterbi(obs[seq])
    print("LLH: ", likelihood)
    print("Re-estimated Model\n", initmodel)

    # Visualize
    uniqueid = "output/diagnostic_plots"
    from matplotlib.pyplot import figure, show
    from matplotlib.backends.backend_pdf import PdfPages

    pp = PdfPages(uniqueid + ".pdf")
    for seq in range(len(obs)):
        fb = figure()
        aa = fb.add_subplot(111)
        x = list(range(obs[seq].shape[1]))
        aa.plot(x, obs[seq].flatten())
        aa.set_xlabel("time (s)")
        aa.set_ylabel("Power (W)")
        aa.set_title("Power Data sequence :" + str(seq))
        fa = figure()
        viz.view_viterbi(fa.add_subplot(1, 1, 1), obs, path, initmodel.O.mu, seq=seq)
        fa.tight_layout()
        pp.savefig()
    show()
    fc = figure()
    viz.view_EMconvergence(fc.add_subplot(1, 1, 1), ll)
    pp.savefig()
    pp.close()
    print("Close the plot window to end the program.")
    show()


if __name__ == "__main__":
    test("local")
