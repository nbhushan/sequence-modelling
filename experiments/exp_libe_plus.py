# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:34:37 2013

@author: nbhushan
"""
import os,sys
dirname=os.path.dirname
sys.path.append(os.path.join(dirname(dirname(__file__)))) 

import numpy as np
from emissionplus import Gaussian
from qdhmm import QDHMM
import hmmviz as viz
import cPickle as pickle


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
    #tau = np.array([0, 120, 720])
    #re-scaled tau's
    tau = np.array([0, 120, 300])
    #tau's obtained from hmm posterior
    #tau = np.array([0, 59, 189])
    p, zeta, eta,  D = .004, .5, .95, np.max(tau)     
    
    # Create an  HMM and Gaussian emmission object        
    emmissionModel = \
        Gaussian(mu = np.array([[834.02 , 340.09, 207.01, 32.71]]), \
                            var = np.array([[[10886.75]], \
                                              [[3733.70]], \
                                              [[2061.09 ]], \
                                              [[85.98]]]), tau=tau)
    libemodel = QDHMM(p, zeta, eta, emmissionModel)

        
    #load libe16099h17h data, this is a real data set    
    numseq = 1
    obs = [np.newaxis] * numseq 
    obs[0]=np.loadtxt("libe16099h17h.csv",delimiter=",",skiprows=1)[np.newaxis,:]
    print 'Length of observation sequence:', obs[0].shape[1]
  

    uniqueid = 'libe16099h17hqdhmm'    
    from matplotlib.pyplot import figure, show   
    from matplotlib.backends.backend_pdf import PdfPages    
    pp = PdfPages(uniqueid+'.pdf')     

    for seq in xrange(len(obs)):
        fb=figure()
        az=fb.add_subplot(1,1,1)
        x=range(obs[seq].shape[1])
        az.plot(x, obs[seq].flatten())
        az.set_title('Data sequence: ' + str(seq) )     
        pp.savefig()        
        pickle.dump(obs[seq], open("Libepowerdata_"+str(seq)+".p",'wb'))

   
    print 'Initial Model \n',libemodel
    
    likelihood,ll = libemodel.qdhmmFit(obs, 10, 1e-5, debug=True)    
    
    path=[None]*len(obs)
    for seq in xrange(len(obs)):
        path[seq] = libemodel.viterbi(obs[seq]) 
        pickle.dump(path[seq], open("Libeviterbipath_"+str(seq)+".p",'wb'))
    
    print 'LLH: ', likelihood
    print 'Re-estimated Model\n',libemodel


   #Visualize
  
    for seq in xrange(len(obs)):
        fa = figure()
        viz.view_viterbi(fa.add_subplot(1,1,1), obs, path, libemodel.O.mu, seq)   
        fa.tight_layout()  
        pp.savefig()

    fc=figure()
    viz.view_EMconvergence(fc.add_subplot(1,1,1),ll)    
    pp.savefig()
    pp.close()    
    print 'Finished! Check results.'  
    #show() 
          
 
if __name__ == '__main__':
    test()