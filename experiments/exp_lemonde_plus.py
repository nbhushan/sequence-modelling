# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:09:13 2013

@author: nbhushan
"""
import os,sys
dirname=os.path.dirname
sys.path.append(os.path.join(dirname(dirname(__file__)))) 

import numpy as np
from emissionplus import Gaussian
from qdhmm import HMMPlus
import time
import hmmviz as viz

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
    np.set_printoptions( precision=4, suppress=True )    
    
    #Initialize model parameters
    tau = np.array([0, 101, 231])
    p, zeta, eta,  D = .004, .8, .7, np.max(tau)     
    
    # Create an  HMM and Gaussian emmission object        
    emmissionModel = \
        Gaussian(mu = np.array([[532.23915 , 270.69044, 76.3893, 26.552]]), \
                            covar = np.array([[[7568.806]], \
                                              [[4568.944]], \
                                              [[247.521 ]], \
                                              [[2.025350]]]), tau=tau)
    lemondemodel = HMMPlus(p, zeta, eta, D, emmissionModel)
    
    #load dauphine110908h19h data, this is a real data set

    numseq = 1
    obs = [np.newaxis] * numseq 
    obs[0]=np.loadtxt("lemonde_test.csv",delimiter=",",skiprows=1)[np.newaxis,:]
    #obs[0] = obs[0][:,1200:11200]
    print 'Length of observation sequence:', obs[0].shape[1]

    
    print 'Initial Model \n',lemondemodel
    start_time=time.time()
    likelihood,ll = lemondemodel.qdhmmFit(obs,10,1e-6,True, 'local')
    end_time=time.time()
    print 'Time taken to estimate parameters (s) :', (end_time-start_time)    
    path=[None]*len(obs)
    for seq in xrange(len(obs)):
        path[seq] = lemondemodel.viterbi(obs[seq]) 
    print 'LLH: ', likelihood
    print 'Re-estimated Model\n',lemondemodel

    
    #Visualize
    uniqueid = '4287'    
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
        viz.view_viterbi(fa.add_subplot(1,1,1), obs, path, lemondemodel.O.mu, seq=seq)   
        fa.tight_layout()  
        pp.savefig()

    fc=figure()
    viz.view_EMconvergence(fc.add_subplot(1,1,1),ll)    
    pp.savefig()
    pp.close()    
    print 'Close the plot window to end the program.'    
    show() 
   
 
if __name__ == '__main__':
    test()