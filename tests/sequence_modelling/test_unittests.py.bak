# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:32:19 2013

@author: nbhushan
"""

import os,sys
dirname=os.path.dirname
sys.path.append(os.path.join(dirname(dirname(__file__)))) 

import unittest2 as unittest
import hmm
import emmissions 
#from hmm_plus import emissionsplus
#from hmm_plus import hmmplus
import logging,sys
import numpy as np


log = logging.getLogger("HMM Unit Tests")
# creating StreamHandler to stderr
hdlr = logging.StreamHandler(sys.stderr)
# setting message format
fmt = logging.Formatter("%(name)s %(filename)s:%(lineno)d  - %(message)s")
hdlr.setFormatter(fmt)
# adding handler to logger object
log.addHandler(hdlr)
#set unittests log level
log.setLevel(logging.ERROR)
#set HMM logging level
hmm.logger.setLevel(logging.ERROR)


class StandardHMMUnitTests(unittest.TestCase):
    
    def setUp(self):
        log.debug("StandardHMMUnitTests.setUp() -- begin")
        self.A = np.array([[0.6, 0.4], [0.6, 0.4],[1./2, 1./2]])    
        self.emmissionModel = emmissions.Gaussian(mu = np.array([[-100., 100.]]), \
                                    covar = np.array([[[ 10.]] ,[[10.]]]))
        self.stdmodel = hmm.StandardHMM(self.A,self.emmissionModel)
        log.debug("HMMBaseClassTests.setUp() -- end")
        
    def testAcessFunctions(self):
        log.debug("StandardHMMUnitTests.testAcessFunctions -- begin")
        
        self.assertEqual(self.stdmodel.K,2)
        np.testing.assert_equal(self.stdmodel.logA, np.log(self.A))
        self.assertEqual(self.stdmodel.logA[1,1], np.log(0.4))
        np.testing.assert_array_equal(np.sum(self.A,1), np.array([1,1,1]))
        self.assertEqual(self.emmissionModel.K, 2)
        self.assertEqual(self.emmissionModel.mu.size, 2)
        self.assertEqual(self.emmissionModel.covar.size, 2)
        log.debug("StandardHMMUnitTests.testAcessFunctions -- end")
        
    def testSample(self):
        log.debug("StandardHMMUnitTests.testSample --begin")
        # single univariate sequence
        N = [100]
        dim = 1
        obs = [np.newaxis] * len(N) 
        for n in range(len(N) ):
            obs[n], zes =self.stdmodel.sample(dim = dim, N =N[n]) 
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].shape[0],1)
        self.assertEqual(obs[0].size, 100 )
        
        # single multivariate sequence
        N = [100]
        dim = 2
        obs = [np.newaxis] * len(N)  
        for n in range(len(N)):
            obs[n], zes =self.stdmodel.sample(dim = dim, N =N[n]) 
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].shape,(2,100))

        
        # two univariate sequences with equal length
        N = [100,100]
        dim = 1
        obs = [np.newaxis] * len(N) 
        for n in range(len(N)):
            obs[n], zes =self.stdmodel.sample(dim = dim, N =N[n]) 
        self.assertEqual(len(obs), 2)
        self.assertEqual(obs[0].shape[0],1)
        self.assertEqual(obs[0].size, 100 )
        self.assertEqual(obs[1].shape[0],1)
        self.assertEqual(obs[1].size, 100 )
        
        # Six univariate sequences with unequal length
        N=[100,200,60,45,600,20]
        dim = 1
        obs = [np.newaxis] * len(N) 
        for n in range(len(N)):
            obs[n], zes =self.stdmodel.sample(dim = dim, N =N[n]) 
        self.assertEqual(len(obs), 6)
        self.assertEqual(obs[0].shape[0],1)
        self.assertEqual(obs[0].size, 100 )
        self.assertEqual(obs[1].shape[0],1)
        self.assertEqual(obs[1].size, 200 )
        self.assertEqual(obs[2].shape[0],1)
        self.assertEqual(obs[2].size, 60 )
        self.assertEqual(obs[3].shape[0],1)
        self.assertEqual(obs[3].size, 45 )
        self.assertEqual(obs[4].shape[0],1)
        self.assertEqual(obs[4].size, 600 )
        self.assertEqual(obs[5].shape[0],1)
        self.assertEqual(obs[5].size, 20 )
        
        # four multivariate sequences with equal length
        N=[100,100,100,100]
        dim = 2
        obs = [np.newaxis] * len(N) 
        for n in range(len(N)):
            obs[n], zes =self.stdmodel.sample(dim = dim, N =N[n]) 
        self.assertEqual(len(obs), 4)
        self.assertEqual(obs[0].shape,(2,100))
        self.assertEqual(obs[1].shape,(2,100))
        self.assertEqual(obs[2].shape,(2,100))
        self.assertEqual(obs[3].shape,(2,100))
        
        # four multivariate sequences with unequal length
        N=[23,30,18,56]
        dim = 2
        obs = [np.newaxis] * len(N) 
        for n in range(len(N)):
            obs[n], zes =self.stdmodel.sample(dim = dim, N =N[n]) 
        self.assertEqual(len(obs), 4)
        self.assertEqual(obs[0].shape,(2,23))
        self.assertEqual(obs[1].shape,(2,30))
        self.assertEqual(obs[2].shape,(2,18))
        self.assertEqual(obs[3].shape,(2,56))
        log.debug("StandardHMMUnitTests.testSample --end")

        
    def testHMMFit(self):
        log.debug("StandardHMMUnitTests.testHMMFit --begin")
        N=[100]
        dim = 1
        obs = [np.newaxis] * len(N) 
        for n in range(len(N)):
            obs[n], zes =self.stdmodel.sample(dim = dim, N =N[n]) 
        self.stdmodel.hmmFit(obs)
        log.debug("StandardHMMUnitTests.testHMMFit --end")
        
       
    def testAlpha(self):
        log.debug("StandardHMMUnitTests.testAlpha --begin")
        obs = [np.newaxis]
        obs[0] = np.loadtxt('univariate_unittest.csv',delimiter = ',')
        obs[0] = obs[0][np.newaxis]
        likelihood, ll, duration, rankn, res = self.stdmodel.hmmFit(obs, maxiter = 1, \
                                 debug=False)
        np.testing.assert_allclose(likelihood, -7.976166,\
                                                    rtol=1e-5, atol=0)
                                                            
        log.debug("StandardHMMUnitTests.testAlpha --end")
       
                                
    def testEM(self):
        log.debug("StandardHMMUnitTests.testEM --begin")
        obs = [np.newaxis]
        obs[0] = np.loadtxt('univariate_unittest.csv',delimiter = ',')
        obs[0] = obs[0][np.newaxis]
        likelihood, ll, duration, rankn, res  = self.stdmodel.hmmFit(obs, 10, 1e-6, \
                                        debug=False)
        np.testing.assert_allclose(np.exp(self.stdmodel.logA),\
                       np.array([[0.5930807, 0.4069193],\
                                 [0.6326531, 0.3673469], \
                                 [0.0, 1.0]]), rtol=1e-5, atol=0)
        np.testing.assert_allclose(self.stdmodel.O.mu, \
                                np.array([[-99.59135, 99.89957]]),\
                                            rtol=1e-5, atol=0)
        np.testing.assert_allclose(self.stdmodel.O.covar, \
                                np.array([[[105.3445]],[[103.5022]]]),\
                                            rtol=1e-5, atol=0)
        assert all(x<=y for x, y in zip(ll, ll[1:]))==True                                          
        log.debug("StandardHMMUnitTests.testEM --end")                                            
        

        
if __name__ == '__main__':
    logging.basicConfig( stream=sys.stderr )
    unittest.main()
        

        
        
        
        
        