# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:32:19 2013
"""

import logging
import sys
import numpy as np
from sequence_modelling.qdhmm import QDHMM
from sequence_modelling.hmm import StandardHMM
from sequence_modelling.emmissions import Gaussian


log = logging.getLogger("HMM Unit Tests")
hdlr = logging.StreamHandler(sys.stderr)
fmt = logging.Formatter("%(name)s %(filename)s:%(lineno)d  - %(message)s")
hdlr.setFormatter(fmt)
log.addHandler(hdlr)
log.setLevel(logging.ERROR)


class TestStandardHMMUnit:
    def setup_method(self):
        log.debug("TestStandardHMMUnit.setup_method() -- begin")
        self.A = np.array([[0.6, 0.4], [0.6, 0.4], [1.0 / 2, 1.0 / 2]])
        self.emmissionModel = Gaussian(
            mu=np.array([[-100.0, 100.0]]), covar=np.array([[[10.0]], [[10.0]]])
        )
        self.stdmodel = StandardHMM(self.A, self.emmissionModel)
        log.debug("TestStandardHMMUnit.setup_method() -- end")

    def test_AccessFunctions(self):
        log.debug("TestStandardHMMUnit.testAccessFunctions -- begin")

        assert self.stdmodel.K == 2
        np.testing.assert_equal(self.stdmodel.logA, np.log(self.A))
        assert self.stdmodel.logA[1, 1] == np.log(0.4)
        np.testing.assert_array_equal(np.sum(self.A, 1), np.array([1, 1, 1]))
        assert self.emmissionModel.K == 2
        assert self.emmissionModel.mu.size == 2
        assert self.emmissionModel.covar.size == 2
        log.debug("TestStandardHMMUnit.testAccessFunctions -- end")

    def test_Sample(self):
        log.debug("TestStandardHMMUnit.testSample -- begin")
        N = [100]
        dim = 1
        obs = [np.newaxis] * len(N)
        for n in range(len(N)):
            obs[n], zes = self.stdmodel.sample(dim=dim, N=N[n])
        assert len(obs) == 1
        assert obs[0].shape[0] == 1
        assert obs[0].size == 100

        # Additional sample tests...

        log.debug("TestStandardHMMUnit.testSample -- end")

    def test_HMMFit(self):
        log.debug("TestStandardHMMUnit.testHMMFit -- begin")
        N = [10000]
        dim = 1
        obs = [np.newaxis] * len(N)
        for n in range(len(N)):
            obs[n], zes = self.stdmodel.sample(dim=dim, N=N[n])
        self.stdmodel.fit(obs)
        log.debug("TestStandardHMMUnit.testHMMFit -- end")

    def test_Alpha(self):
        log.debug("TestStandardHMMUnit.testAlpha -- begin")
        obs = [np.newaxis]
        obs[0] = np.loadtxt(
            "tests/sequence_modelling/data/univariate_unittest.csv", delimiter=","
        )
        obs[0] = obs[0][np.newaxis]
        likelihood, ll, duration, rankn, res = self.stdmodel.fit(
            obs, maxiter=1, debug=False
        )
        np.testing.assert_allclose(likelihood, -7.976166, rtol=1e-5, atol=0)

        log.debug("TestStandardHMMUnit.testAlpha -- end")

    def test_EM(self):
        log.debug("TestStandardHMMUnit.testEM -- begin")
        obs = [np.newaxis]
        obs[0] = np.loadtxt(
            "tests/sequence_modelling/data/univariate_unittest.csv", delimiter=","
        )
        obs[0] = obs[0][np.newaxis]
        likelihood, ll, duration, rankn, res = self.stdmodel.fit(
            obs, 10, 1e-6, debug=False
        )
        np.testing.assert_allclose(
            np.exp(self.stdmodel.logA),
            np.array([[0.5930807, 0.4069193], [0.6326531, 0.3673469], [0.0, 1.0]]),
            rtol=1e-5,
            atol=0,
        )
        np.testing.assert_allclose(
            self.stdmodel.O.mu, np.array([[-99.59135, 99.89957]]), rtol=1e-5, atol=0
        )
        np.testing.assert_allclose(
            self.stdmodel.O.covar,
            np.array([[[105.3445]], [[103.5022]]]),
            rtol=1e-5,
            atol=0,
        )
        assert all(x <= y for x, y in zip(ll, ll[1:])) == True
        log.debug("TestStandardHMMUnit.testEM -- end")
