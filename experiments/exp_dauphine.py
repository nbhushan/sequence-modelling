# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:34:37 2013

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
    emmissionModel = \
        Gaussian(mu = np.array([[1054.39 , 329.88, 154.60, 52.36]]), \
                            covar = np.array([[[46560.05]], \
                                              [[2050.68]], \
                                              [[2734.04 ]], \
                                              [[159.58]]]))
    dauphinemodel = StandardHMM(A, emmissionModel)
    
    #load dauphine110908h19h data, this is a real data set
    
    numseq = 1
    obs = [np.newaxis] * numseq 
    obs[0]= np.loadtxt("dauphine11071309h15h.csv",delimiter=",",skiprows=1)[np.newaxis,:]
    #obs[1] = np.loadtxt("hmm_plus\\dauphine15071309h15h.csv",delimiter=",",skiprows=1)[np.newaxis,:]
    #obs[0] = obs[0][:,6000:14000]
    #Re-sample every 10 s
    obs[0]=np.mean(obs[0].flatten().reshape(-1, 2), axis=1)[np.newaxis,:]
    #obs[1]=np.mean(obs[1].flatten().reshape(-1, 2), axis=1)[np.newaxis,:]
    print 'Length of observation sequence 1 :', obs[0].shape[1]
    #print 'Length of observation sequence 2 :', obs[1].shape[1]
    

    
    print 'Initial Model \n',dauphinemodel
    print 'Initial Emission Model\n', dauphinemodel.O
    
    likelihood,ll, durationlist, ranknlist, reslist= dauphinemodel.hmmFit(obs,20,1e-6, debug=True)   
    path=[None]*numseq
    for seq in xrange(numseq):
        path[seq] = dauphinemodel.viterbi(obs[seq]) 
    
    print 'LLH: ', likelihood
    print 'Re-estimated Model\n',dauphinemodel
    print 'Re-estimated Emission Model\n',dauphinemodel.O
    
    #Viterbi state duration
    lengths=[None]*numseq
    for idx,stateseq in enumerate(path):
        lengths[idx] = dauphinemodel.estimateviterbiduration(stateseq)
    for length in lengths:        
        print 'Viterbi state duration:', np.max(length[0]) ,np.max(length[1]), np.max(length[2]), np.max(length[3])     
    for dur in durationlist:
        print 'Posterior distribution duration estimation:', dur
    
    #Visualize
    #filepath = 'C:\\Local\\FINALEXPERIMENTS\\'
    uniqueid = 'dauphine_stdhmm'    
    from matplotlib.pyplot import figure, show   
    from matplotlib.backends.backend_pdf import PdfPages    
    #pp = PdfPages(filepath+uniqueid+'.pdf')      
    pp = PdfPages(uniqueid+'.pdf')
    fa = figure()
    viz.view_viterbi(fa.add_subplot(1,1,1), obs, path, dauphinemodel.O.mu, seq=0)   
    fa.tight_layout()  
    pp.savefig()
    fb=figure()
    viz.view_postduration(fb.add_subplot(111), obs, path, dauphinemodel.O.mu, reslist, ranknlist, seq=0)
    fb.tight_layout()
    pp.savefig()
    fc=figure()
    viz.view_EMconvergence(fc.add_subplot(1,1,1),ll)    
    pp.savefig()
    pp.close()    
    print 'Close the plot window to end the program.'    
    show() 
       
 
if __name__ == '__main__':
    test()