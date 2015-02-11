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
    tau = np.array([0, 60, 360])
    #tau's obtained from hmm posterior
    #tau = np.array([0, 59, 189])
    p, zeta, eta,  D = .004, .4, .99, np.max(tau)     
    
    # Create an  HMM and Gaussian emmission object        
    emmissionModel = \
        Gaussian(mu = np.array([[1254.39 , 429.88, 254.60, 52.36]]), \
                            var = np.array([[[46560.05]], \
                                              [[7500.68]], \
                                              [[2734.04 ]], \
                                              [[159.58]]]), tau=tau)
    dauphinemodel = QDHMM(p, zeta, eta, emmissionModel)

    '''    
    #load dauphine110908h19h data, this is a real data set    
    numseq = 1
    obs = [np.newaxis] * numseq 
    obs[0]=np.loadtxt("dauphine15071309h15h.csv",delimiter=",",skiprows=1)[np.newaxis,:]
    #obs[0] = obs[0][:,6000:14000]
    #Re-sample every 10 s
    obs[0]=np.mean(obs[0].flatten().reshape(-1, 2), axis=1)[np.newaxis,:]
    print 'Length of observation sequence:', obs[0].shape[1]
    
    
    import matplotlib.pyplot as plt
    x=range(obs[0].shape[1])
    plt.plot(x,obs[0].flatten())
    plt.show()
    '''
    temp=[]
    test =np.loadtxt("dauphine_1107to1108_powerdata.csv",delimiter=",",skiprows=1)
    from itertools import islice
    i = iter(test)
    piece = list(islice(i, 43200))
    while piece:
        temp.append(piece)
        piece = list(islice(i, 43200))    
        
    obs=[np.array(x)[np.newaxis,:] for x in temp]
    for x,seq in enumerate(obs):
        obs[x] = np.mean(seq.flatten().reshape(-1, 2), axis=1)[np.newaxis,:]
    
    #Filter observation sequences based on Variance
    means=[]
    var=[]
    x,y=0,0
    for seq in range(len(obs)):
        print 'seq : ', seq
        y = obs[seq].shape[1] + y
        print x , y
        x=y
        print 'Mean: ', np.mean(obs[seq])
        means.append(np.mean(obs[seq]))
        print 'Variance : ', np.var(obs[seq])
        var.append(np.var(obs[seq]))
        print '-'*60
    

    uniqueid = 'dauphine_1107to1109qdhmm'    
    from matplotlib.pyplot import figure, show   
    from matplotlib.backends.backend_pdf import PdfPages    
    pp = PdfPages(uniqueid+'.pdf')     
    
    data = np.mean(test.flatten().reshape(-1, 2), axis=1) 
    fb=figure()
    az=fb.add_subplot(1,1,1)
    az.plot(data)
    az.set_title('Data')     
    pp.savefig()        
    
    fa = figure()
    ax = fa.add_subplot(121)    
    ay = fa.add_subplot(122)    
    ax.plot(means, label='Mean')
    ay.plot(var, label='Variance')
    ax.legend()
    ay.legend()
    ax.set_title('Mean of individual sequences')
    ax.set_xlabel('seq ')
    ax.set_ylabel('Mean (W)')    
    ay.set_title('Variance of individual sequences')
    ay.set_xlabel('seq ')
    ay.set_ylabel('Variance')    
    pp.savefig()                  
    #show()
    
    data = [seq for seq in obs if np.var(seq)> 4000.]
    
    print 'number of sequences: ', len(data)
    for seq in range(len(data)):
        print 'Length of observation sequence ' + str(seq) + ' :', data[seq].shape[1]    
    
    print 'Initial Model \n',dauphinemodel
    
    likelihood,ll = dauphinemodel.qdhmmFit(data,5,1e-5, debug=True)    
    
    path=[None]*len(data)
    for seq in xrange(len(data)):
        path[seq] = dauphinemodel.viterbi(data[seq]) 
    
    print 'LLH: ', likelihood
    print 'Re-estimated Model\n',dauphinemodel


   #Visualize
  
    for seq in xrange(len(data)):
        fa = figure()
        viz.view_viterbi(fa.add_subplot(1,1,1), data, path, dauphinemodel.O.mu, seq)   
        fa.tight_layout()  
        pp.savefig()

    fc=figure()
    viz.view_EMconvergence(fc.add_subplot(1,1,1),ll)    
    pp.savefig()
    pp.close()    
    print 'Close the plot window to end the program.'    
    #show() 
          
 
if __name__ == '__main__':
    test()