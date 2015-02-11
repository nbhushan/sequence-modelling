# -*- coding: utf-8 -*-
"""
Created on Thu Aug 08 19:42:28 2013

@author: nbhushan
"""
import os,sys
dirname=os.path.dirname
sys.path.append(os.path.join(dirname(dirname(__file__)))) 

import numpy as np
#from scipy.cluster.vq import *
from emmissions import Gaussian
from hmm import StandardHMM
import emissionplus 
from qdhmm import QDHMM
import hmmviz as viz
import time
#import pdb
import logging
logging.basicConfig(filename='C:\\Local\\thesis\\experiments\\SAsearch\\dauphinepipeline1.log',level=logging.DEBUG)

def _generatedata(N, nseq, dim, tau,zeta, eta):
    """
    This function has the following workflow:
        1) Define an initial gaussian emmission object
        2) Generate observations from the emmission object
        3) Define intial A , where pi = A[-1]
        4) Train the model and find params which best fit the observations
        5) Visualize the state sequence.
    """
                      
    #Initialize model parameters
    tau = tau    
    p, zeta, eta,  D = .001, zeta, eta, np.max(tau)     
    
    # Create an  HMM and Gaussian emmission object to sample data       
    _emmissionModel = \
          emissionplus.Gaussian(mu = np.array([[1254.39 , 429.88, 254.60, 52.36]]), \
                            var = np.array([[[46560.05]], \
                                              [[2500.68]], \
                                              [[734.04 ]], \
                                              [[159.58]]]),tau=tau)
    samplemodel = QDHMM(p, zeta, eta, _emmissionModel)
    
    #sample data from he emmission model
        
    N = [N]
    dim = dim
    numseq = nseq
    obs = [np.newaxis] * numseq 

    for n in range(numseq):
        obs[n],zes = samplemodel.sample(dim , N[n]) 
   
    return obs        
        
def fitHMM(obs):
    '''
    This function is used to train a HMM on the data and obtain 
    initial estimates for the QDHMM
    TODO: Write K-means code to obtain initial emission model.
    '''    
    #print obs[0].T.shape
    #mean = kmeans2(obs[0].T, k=4, iter = 15, minit='random')
    #print mean    
    
    #Initialize transition Matrix
    A = np.array([[0.3832979, 0.6167021, 0.0, 0.0 ], \
                  [0.3507986, 0.3821364, 0.2670650, 0.0],\
                  [0.3494203, 0.0, 0.3399672, 0.3106124],\
                  [0.2237589, 0.0, 0.0, 0.7762411],\
                  [1./4, 1./4, 1./4, 1./4 ]])   
    

    #Dauphine emission model                                              
    _emmissionModel = \
          Gaussian(mu = np.array([[1254.39 , 429.88, 254.60, 52.36]]), \
                            covar = np.array([[[46560.05]], \
                                              [[5500.68]], \
                                              [[2734.04 ]], \
                                              [[159.58]]]))                                                 
    hmmmodel = StandardHMM(A, _emmissionModel)    
    # Fit the model to the data and print results
    start_time=time.time()
    newloglikehood,ll, duration, rankn, res = hmmmodel.hmmFit(obs , maxiter = 10 , epsilon = 1e-6, \
                                     debug=True)  
    end_time=time.time()
    logging.info('Time to estimate initial HMM model : %.5f s '% (end_time-start_time))                             
    return (hmmmodel, duration)
    
def fitQDHMM(obs, initmodel, tau):
    '''
    This function fits a QDHMM to the data using the initial parameters 
    obtained from HMM
    '''
    #pdb.set_trace()
    #assign zeta and eta at random
    #_tau = np.array([0,60,360])
    #_p, _zeta, _eta = .001, .3, .9
    #use HMM to initialize QDHMM
    A = np.exp(initmodel.logA)
    _p = A[-1,0]
    _zeta = A[0,0]
    _eta = np.mean (np.diag(A)[1:])    
    _emmissionModel = \
        emissionplus.Gaussian(mu = initmodel.O.mu, var = initmodel.O.covar, tau=tau)  
    '''                                           
    #Dauphine emission model                                              
    _emmissionModel = \
        emissionplus.Gaussian(mu = np.array([[1254.39 , 429.88, 254.60, 52.36]]), \
                            covar = np.array([[[46560.05]], \
                                              [[7500.68]], \
                                              [[2734.04 ]], \
                                              [[159.58]]]), tau=tau)  
    '''
    qdhmmmodel = QDHMM(_p, _zeta, _eta, _emmissionModel)        
    '''
    #train the model in the data and print results
    logging.info('Initial Model used to initialize EM : %.s ' % qdhmmmodel )     
    print 'Initial Model used to initialize EM  \n',qdhmmmodel 
    logging.info('Initial Emission Model obtained by standard HMM: %.s ' % qdhmmmodel.O )     
    print 'Initial Emission Model obtained by standard HMM\n', qdhmmmodel.O
    '''
    start_time=time.time()
    likelihood,ll = qdhmmmodel.qdhmmFit(obs, 10, 1e-6, True, 'local')
    end_time=time.time()
    logging.info('Time taken to estimate QDHMM parameters (s) : %.5f s' % (end_time-start_time) )  
    print 'Time taken to estimate QDHMM parameters (s) :', (end_time-start_time)
    path=[None]*len(obs)
    for seq in xrange(len(obs)):
        path[seq] = initmodel.viterbi(obs[seq]) 
    logging.info('LLH: : %.5f s' % likelihood ) 
    print 'LLH: ', likelihood
    logging.info('Re-estimated Model : %.s ' % qdhmmmodel )   
    print 'Re-estimated Model\n',qdhmmmodel
    logging.info('Re-estimated Emission Model: %.s ' % qdhmmmodel.O )  
    print 'Re-estimated Emission Model\n',qdhmmmodel.O
    
    #Visualize
    uniqueid = 'finalpipelinetest'    
    from matplotlib.pyplot import figure, show   
    from matplotlib.backends.backend_pdf import PdfPages    
    pp = PdfPages(uniqueid+'.pdf')  
    for seq in range(len(obs)):    
        fb = figure()
        aa = fb.add_subplot(111)
        x=range(obs[seq].shape[1])
        aa.plot(x,obs[seq].flatten())
        aa.set_xlabel('time (s)')
        aa.set_ylabel('Power (W)')
        aa.set_title('Power Data sequence :' + str(seq))
        fa = figure()
        viz.view_viterbi(fa.add_subplot(1,1,1), obs, path, initmodel.O.mu, seq=seq)   
        fa.tight_layout()  
        pp.savefig()

    fc=figure()
    viz.view_EMconvergence(fc.add_subplot(1,1,1),ll)    
    pp.savefig()
    pp.close()    
    print 'Close the plot window to end the program.'    
    show()  
    
if __name__=='__main__':
    # adjust the precision of printing float values
    np.set_printoptions( precision=4, suppress=True )        
    
    #Sample Data
    obs = _generatedata(N=1000, nseq=1, dim=1, tau=np.array([0,6,10]),zeta=0.65, eta=0.966)
    
    '''
    # Dauphine data    
    numseq = 1
    obs = [np.newaxis] * numseq 
    obs[0]=np.loadtxt("dauphine11071309h15h.csv",delimiter=",",skiprows=1)[np.newaxis,:]
    #Re-sample every 2 s
    obs[0]=np.mean(obs[0].flatten().reshape(-1, 2), axis=1)[np.newaxis,:]
    '''
    
    print 'Length of observation sequence:', obs[0].shape[1]    
    
   # Estimate initial parameters using HMM
    initmodel, pstdur = fitHMM(obs)
    print initmodel
    taus = np.array(pstdur).flatten()
    print 'duration: ', np.array(pstdur).flatten()
    #pstdur = np.array([0,29,93])
    #Fit QDHMM to data
    fitQDHMM(obs, initmodel, np.array([0, taus[1], taus[1]+taus[2]]))