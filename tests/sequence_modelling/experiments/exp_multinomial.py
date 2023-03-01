# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:05:45 2013

@author: nbhushan
"""
import numpy as np
from emmissions import Discrete
from hmm import StandardHMM

def test():
   
    # adjust the precision of printing float values
    np.set_printoptions( precision=4, suppress=True )    
    
    # initial transition matrix
    A = np.array([[.7, .3],[.2, .8],[.5, .5]])   

    #initial probability vector 
    #where p_i is the probability of seeing outcome i      
    c = [0,1,2,3,4,5]
    p = np.array([[1./6, 1./6, 1./6, 1./6, 1./6, 1./6], [.1 ,.1,.1, .1, .1, .5]])
    
    # Create an  HMM and Multinomial emmission object        
    emmissionModel = Discrete(p,c)
    model = StandardHMM(A,emmissionModel)
   
    #sample data from he emmission model
    N = [1000]
    dim = 1
    numseq =1
    obs = [np.newaxis] * numseq 
    for n in range(numseq):
        obs[n] = model.sample(dim = dim, N =N[n]) 
    
    #np.savetxt('discrete_discasino.csv', np.transpose(obs[0]), fmt='%.2f' ,delimiter = ',')
    
             
    print("\nHMM MODEL:\n ")    
    print(model)
    print()
    print("\nOriginal Emmission model: \n" , emmissionModel)
    print('*'*80)   
    
    # Fit the model to the data and print results
    newloglikehood = model.hmmFit(obs , maxiter = 12 , epsilon = 0.00001)    
    (path, V, psi)  = model.viterbi(obs[0])    
    
    print("Log likelihood: \n" , newloglikehood)  
    print("Re-estimated HMM Model: \n" , model)
    print("Re-estimated Emmission model: \n" , model.O)
   
    from matplotlib.pyplot import figure, show 
    
    fa = figure()
    model.displayviterbi(fa.add_subplot(1,1,1),*model.formatdata(obs[0][0],path,np.array([1,4])))
    fa.tight_layout()
    
    '''
    fb = figure()
    model.visTraining(ll,fb.add_subplot(1,1,1))    
    fb.tight_layout()
    '''
    print('Close the plot window to end the program.')        
    show()
    
 
if __name__ == '__main__':
    test()