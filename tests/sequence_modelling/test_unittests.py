# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:32:19 2013

@author: nbhushan
"""

import unittest2 as unittest
from sequence_modelling.hmm import StandardHMM
from sequence_modelling.emmissions import Gaussian
import logging, sys
import numpy as np


log = logging.getLogger("HMM Unit Tests")
# creating StreamHandler to stderr
hdlr = logging.StreamHandler(sys.stderr)
# setting message format
fmt = logging.Formatter("%(name)s %(filename)s:%(lineno)d  - %(message)s")
hdlr.setFormatter(fmt)
# adding handler to logger object
log.addHandler(hdlr)
# set unittests log level
log.setLevel(logging.ERROR)
# set HMM logging level
# hmm.logger.setLevel(logging.ERROR)


class StandardHMMUnitTests(unittest.TestCase):
    def setUp(self):
        log.debug("StandardHMMUnitTests.setUp() -- begin")
        self.A = np.array([[0.6, 0.4], [0.6, 0.4], [1.0 / 2, 1.0 / 2]])
        self.emmissionModel = Gaussian(
            mu=np.array([[-100.0, 100.0]]), covar=np.array([[[10.0]], [[10.0]]])
        )
        self.stdmodel = StandardHMM(self.A, self.emmissionModel)
        log.debug("HMMBaseClassTests.setUp() -- end")

    def test_AcessFunctions(self):
        log.debug("StandardHMMUnitTests.testAcessFunctions -- begin")

        assert self.stdmodel.K == 2
        np.testing.assert_equal(self.stdmodel.logA, np.log(self.A))
        assert self.stdmodel.logA[1, 1] == np.log(0.4)
        np.testing.assert_array_equal(np.sum(self.A, 1), np.array([1, 1, 1]))
        assert self.emmissionModel.K == 2
        assert self.emmissionModel.mu.size == 2
        assert self.emmissionModel.covar.size == 2
        log.debug("StandardHMMUnitTests.testAcessFunctions -- end")

    def test_Sample(self):
        log.debug("StandardHMMUnitTests.testSample --begin")
        # single univariate sequence
        N = [100]
        dim = 1
        obs = [np.newaxis] * len(N)
        for n in range(len(N)):
            obs[n], zes = self.stdmodel.sample(dim=dim, N=N[n])
        assert len(obs) == 1
        assert obs[0].shape[0] == 1
        assert obs[0].size == 100

        # single multivariate sequence
        N = [100]
        dim = 2
        obs = [np.newaxis] * len(N)
        for n in range(len(N)):
            obs[n], zes = self.stdmodel.sample(dim=dim, N=N[n])
        assert len(obs) == 1
        assert obs[0].shape == (2, 100)

        # two univariate sequences with equal length
        N = [100, 100]
        dim = 1
        obs = [np.newaxis] * len(N)
        for n in range(len(N)):
            obs[n], zes = self.stdmodel.sample(dim=dim, N=N[n])
        assert len(obs) == 2
        assert obs[0].shape[0] == 1
        assert obs[0].size == 100
        assert obs[1].shape[0] == 1
        assert obs[1].size == 100

        # Six univariate sequences with unequal length
        N = [100, 200, 60, 45, 600, 20]
        dim = 1
        obs = [np.newaxis] * len(N)
        for n in range(len(N)):
            obs[n], zes = self.stdmodel.sample(dim=dim, N=N[n])
        assert len(obs) == 6
        assert obs[0].shape[0] == 1
        assert obs[0].size == 100
        assert obs[1].shape[0] == 1
        assert obs[1].size == 200
        assert obs[2].shape[0] == 1
        assert obs[2].size == 60
        assert obs[3].shape[0] == 1
        assert obs[3].size == 45
        assert obs[4].shape[0] == 1
        assert obs[4].size == 600
        assert obs[5].shape[0] == 1
        assert obs[5].size == 20

        # four multivariate sequences with equal length
        N = [100, 100, 100, 100]
        dim = 2
        obs = [np.newaxis] * len(N)
        for n in range(len(N)):
            obs[n], zes = self.stdmodel.sample(dim=dim, N=N[n])
        assert len(obs) == 4
        assert obs[0].shape == (2, 100)
        assert obs[1].shape == (2, 100)
        assert obs[2].shape == (2, 100)
        assert obs[3].shape == (2, 100)

        # four multivariate sequences with unequal length
        N = [23, 30, 18, 56]
        dim = 2
        obs = [np.newaxis] * len(N)
        for n in range(len(N)):
            obs[n], zes = self.stdmodel.sample(dim=dim, N=N[n])
        assert len(obs) == 4
        assert obs[0].shape == (2, 23)
        assert obs[1].shape == (2, 30)
        assert obs[2].shape == (2, 18)
        assert obs[3].shape == (2, 56)
        log.debug("StandardHMMUnitTests.testSample --end")

    def test_HMMFit(self):
        log.debug("StandardHMMUnitTests.testHMMFit --begin")
        N = [10000]
        dim = 1
        obs = [np.newaxis] * len(N)
        for n in range(len(N)):
            obs[n], zes = self.stdmodel.sample(dim=dim, N=N[n])
        self.stdmodel.hmmFit(obs)
        log.debug("StandardHMMUnitTests.testHMMFit --end")

    def test_Alpha(self):
        log.debug("StandardHMMUnitTests.testAlpha --begin")
        obs = [np.newaxis]
        obs[0] = np.loadtxt(
            "tests/sequence_modelling/data/univariate_unittest.csv", delimiter=","
        )
        obs[0] = obs[0][np.newaxis]
        likelihood, ll, duration, rankn, res = self.stdmodel.hmmFit(
            obs, maxiter=1, debug=False
        )
        np.testing.assert_allclose(likelihood, -7.976166, rtol=1e-5, atol=0)

        log.debug("StandardHMMUnitTests.testAlpha --end")

    def test_EM(self):
        log.debug("StandardHMMUnitTests.testEM --begin")
        obs = [np.newaxis]
        obs[0] = np.loadtxt(
            "tests/sequence_modelling/data/univariate_unittest.csv", delimiter=","
        )
        obs[0] = obs[0][np.newaxis]
        likelihood, ll, duration, rankn, res = self.stdmodel.hmmFit(
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
        log.debug("StandardHMMUnitTests.testEM --end")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    unittest.main()
