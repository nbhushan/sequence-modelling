# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:29:21 2013

@author: nbhushan
"""
import os,sys
dirname=os.path.dirname
sys.path.append(os.path.join(dirname(dirname(__file__)))) 

import numpy as np
from emmissions import Gaussian
from hmm import StandardHMM
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
    
    #Initialize transition Matrix
    A = np.array([[0.3832979, 0.6167021, 0.0, 0.0 ], \
                  [0.3507986, 0.3821364, 0.2670650, 0.0],\
                  [0.3494203, 0.0, 0.3399672, 0.3106124],\
                  [0.2237589, 0.0, 0.0, 0.7762411],\
                  [1./4, 1./4, 1./4, 1./4 ]])   

    
    # Create an  HMM and Gaussian emmission object        
    emmissionModel = Gaussian(mu = np.array([[532.2391, 270.6904, 76.389, 26.552]]),\
                                covar = np.array([[[ 7568.7]] ,\
                                                  [[ 4568.94]],\
                                                  [[ 247.521]],\
                                                  [[ 2.025  ]]]))
    model = StandardHMM(A,emmissionModel)
    
    numseq = 1
    obs = [np.newaxis] * numseq 
    obs[0]=np.loadtxt(r"lemonde230708h19h.csv",delimiter=",",skiprows=1)[np.newaxis,:]
    #obs[0] = obs[0][:,1200:11200]
    print 'Length of observation sequence:', obs[0].shape[1]
 
            

    print "\nHMM MODEL USED TO INITIALIZE EM:\n "    
    print model
    print '*'*80   

    # Fit the model to the data and print results
    newloglikehood,ll, duration, rankn, res = model.hmmFit(obs , maxiter = 2 , epsilon = 1e-6, \
                                     debug=True)    
    path=[None]*numseq
    for seq in xrange(numseq):
        path[seq] = model.viterbi(obs[seq])    
    
    print "Log likelihood: \n" , newloglikehood  
    print "Re-estimated HMM Model: \n" , model
    print 'posterior taus:', duration
    #Viterbi state duration
    lengths=[None]*numseq
    for idx,stateseq in enumerate(path):
        lengths[idx] = model.estimateviterbiduration(stateseq)
    for length in lengths:        
        print 'Viterbi state duration:', np.max(length[0]) ,np.max(length[1]), np.max(length[2]), np.max(length[3])     
    print 'Posterior distribution duration estimation:', duration
    
    #Visualize
    uniqueid = 'lemondeqdhmm'
    from matplotlib.pyplot import figure, show    
    from matplotlib.backends.backend_pdf import PdfPages    
    pp = PdfPages(uniqueid+'.pdf')      
    for seq in xrange(numseq):
        fa = figure()
        viz.view_viterbi(fa.add_subplot(1,1,1), obs, path, model.O.mu, seq)   
        fa.tight_layout()    
        pp.savefig()
        fb=figure()
        viz.view_postduration(fb.add_subplot(111), obs, path, model.O.mu, res, rankn, seq)
        fb.tight_layout()
        pp.savefig()
    fc=figure()
    viz.view_EMconvergence(fc.add_subplot(1,1,1),ll)        
    print 'Close the plot window to end the program.'  
    pp.savefig()
    show() 
 
 
if __name__ == '__main__':
    test()