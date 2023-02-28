# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:34:37 2013

@author: nbhushan
"""

import os,sys, csv
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
    A = np.array([[1./5, 1./5, 1./5, 1./5, 1./5], \
                  [1./5, 1./5, 1./5, 1./5, 1./5],\
                  [1./5, 1./5, 1./5, 1./5, 1./5],\
                  [1./5, 1./5, 1./5, 1./5, 1./5], \
                  [1./5, 1./5, 1./5, 1./5, 1./5], \
                  [1./5, 1./5, 1./5, 1./5, 1./5]])   
    # Create an  HMM and Gaussian emmission object        
    emmissionModel = \
        Gaussian(mu = np.array([[3902.85 , 2332.44, 809.70, 258.08, 76.70]]), \
                            covar = np.array([[[341.05]], \
                                              [[215.68]], \
                                              [[189.58]], \
                                              [[90.32]], \
                                              [[47.44]]]))
    wintermodel = StandardHMM(A, emmissionModel)
    
    #load dauphine110908h19h data, this is a real data set
    
    numseq = 1
    obs = [np.newaxis] * numseq 
    obs[0]= np.loadtxt("1Household.csv",delimiter=",",skiprows=1)[np.newaxis,:]
    print('Length of observation sequence 2 :', obs[0].shape[1])
    

    
#==============================================================================
    print('Initial Model \n',wintermodel)
    print('Initial Emission Model\n', wintermodel.O)
    
    likelihood,ll, durationlist, ranknlist, reslist= wintermodel.hmmFit(obs,100,1e-6, debug=True)   
    path=[None]*numseq
    for seq in range(numseq):
        path[seq] = wintermodel.viterbi(obs[seq]) 
    
    print('LLH: ', likelihood)
    print('Re-estimated Model\n',wintermodel)
    print('Re-estimated Emission Model\n',wintermodel.O)
#==============================================================================
    
    print("Creating 50 energy profiles of feeder A")
    profile_list =[]
    n = 0
    while n < 50:
        profile_list.append(wintermodel.sample(N = 1440)[0][0])
        n = n+1
    new_list = list(map(list, list(zip(*profile_list))))     
    feeder_power_A = sum(new_list,1).T

#==============================================================================
#==============================================================================
    
    print("Creating 50 energy profiles of feeder B")
    profile_list =[]
    n = 0
    while n < 50:
        profile_list.append(wintermodel.sample(N = 1440)[0][0])
        n = n+1
    new_list = list(map(list, list(zip(*profile_list))))     
    feeder_power_B = sum(new_list,1).T

#==============================================================================
#==============================================================================
    
    print("Creating 50 energy profiles of feeder C")
    profile_list =[]
    n = 0
    while n < 50:
        profile_list.append(wintermodel.sample(N = 1440)[0][0])
        n = n+1
    new_list = list(map(list, list(zip(*profile_list))))     
    feeder_power_C = sum(new_list,1).T

#==============================================================================
#==============================================================================
    
    print("Creating 50 energy profiles of feeder D")
    profile_list =[]
    n = 0
    while n < 50:
        profile_list.append(wintermodel.sample(N = 1440)[0][0])
        n = n+1
    new_list = list(map(list, list(zip(*profile_list))))     
    feeder_power_D = sum(new_list,1).T

#==============================================================================
    feeder_power = np.column_stack((feeder_power_A, feeder_power_B, feeder_power_C, feeder_power_D))

    np.savetxt('feeder_power.csv', feeder_power, delimiter=',',fmt='%f') 
    #with open("HMM_profiles_feederA.csv", "wb") as f:
        #writer = csv.writer(f, delimiter = ',')
        #writer.writerows(test)
#==============================================================================
            
    
    
#    #Viterbi state duration
#    lengths=[None]*numseq
#    for idx,stateseq in enumerate(path):
#        lengths[idx] = wintermodel.estimateviterbiduration(stateseq)
#    for length in lengths:        
#        print 'Viterbi state duration:', np.max(length[0]) ,np.max(length[1]), np.max(length[2]), np.max(length[3])     
#    for dur in durationlist:
#        print 'Posterior distribution duration estimation:', dur
#    
#    #Visualize
#    #filepath = 'C:\\Local\\FINALEXPERIMENTS\\'
#    uniqueid = 'winter_stdhmm'    
#    from matplotlib.pyplot import figure, show   
#    from matplotlib.backends.backend_pdf import PdfPages    
#    #pp = PdfPages(filepath+uniqueid+'.pdf')      
#    pp = PdfPages(uniqueid+'.pdf')
#    fa = figure()
#    viz.view_viterbi(fa.add_subplot(1,1,1), obs, path, wintermodel.O.mu, seq=0)   
#    fa.tight_layout()  
#    pp.savefig()
#    
#    fb=figure()
#    viz.view_postduration(fb.add_subplot(111), obs, path, wintermodel.O.mu, reslist, ranknlist, seq=0)
#    fb.tight_layout()
#    pp.savefig()
#    
#    fc=figure()
#    viz.view_EMconvergence(fc.add_subplot(1,1,1),ll)    
#    pp.savefig()
#    pp.close()    
#    print 'Close the plot window to end the program.'    
#    show() 
       
 
if __name__ == '__main__':
    test()