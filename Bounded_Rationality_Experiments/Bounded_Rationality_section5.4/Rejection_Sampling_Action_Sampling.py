#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:04:55 2019

@author: defetro
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:11:51 2019

@author: defetro
"""

import numpy as np
import math as m
import seaborn as sns
import itertools
import matplotlib
import theano
import itertools
import pymc3 as pm
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy as sp


def transition_matrix(transitions,n_action_state):
    """
    Building a transition matrix from the underling vector of parameter given by the betabinomial
    """
    n = n_action_state #number of states and actios

    M=[[0 for x in range(n)] for y in range(n)]
    #each entry in the matrix represent a probability distribution of going

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]

    return M

def sample_Beta_Binomial(a, b, n, size=None):
    """Sampling from a Beta Binomial"""
    p = np.random.beta(a, b, size=size)
    r = np.random.binomial(n, p)
    return r


def normalization(p_betabino,n,N):
    """Normalizing Function for Beta Binomial"""
    p = np.zeros(n, dtype=np.float64) # histogram
    for v in p_betabino: # fill it
        p[v] += 1.0

    p /= np.float64(N)#normalization

    return p

def rejection_sampling(utility_matrix,a,state,beta_value,d_prior,states,
                action_list,utilities_action,t_m,costs,orizon,o,p_true):

    """Variant of Metropolis Algorithm with Utility as Likelihood."""
    #sample a random number between 0 and 1
    u=np.random.uniform(0,1,1)
    
    #sample a new action from the prior
    a_prime=np.random.choice(states,1, p=d_prior)
    
    #new state given new action according to transtion Matrix
    state_new=np.random.choice(states,1, p=p_true)
    
    #difference current utility to pre-specified target
    
    delta=(utility_matrix[a_prime[0]][state_new])-1

                                
    #exponent of the modifier
    weight=delta*beta_value
    
    #acceptance probability
    acceptance_p= np.exp(weight)
    
    
    if u >= acceptance_p :
         #run the chain still
         
         rejection_sampling(utility_matrix,a,state,beta_value,d_prior,states,
                action_list,utilities_action,t_m,costs,orizon,o,p_true)
    
    
    elif acceptance_p >= u :

        action_list.append(a_prime[0])
        utilities_action.append(utility_matrix[a_prime[0]][state_new])

        return action_list,utilities_action,costs




def boltzman_d(u,beta,d_prior):
    """Compute equilibrium distribution where the costs are implicitly given bt the temperature beta"""
    e_x = d_prior*np.exp(u*beta)
    return e_x / e_x.sum(axis=0) # only difference



def bootstrap(data, n=1000, func=np.mean):
    """
    Generate `n` bootstrap samples, evaluating `func`
    at each resampling. `bootstrap` returns a function,
    which can be called to obtain confidence intervals
    of interest.
    """
    simulations = list()
    sample_size = len(data)
    xbar_init = np.mean(data)
    for c in range(n):
        itersample = np.random.choice(data, size=sample_size, replace=True)
        simulations.append(func(itersample))
    simulations.sort()
    def ci(p):
        """
        Return 2-sided symmetric confidence interval specified
        by p.
        """
        u_pval = (1+p)/2.
        l_pval = (1-u_pval)
        l_indx = int(np.floor(n*l_pval))
        u_indx = int(np.floor(n*u_pval))
        return(simulations[l_indx],simulations[u_indx])
    return(ci)



def actions(n_action_state,states,k_samples,p_true,utility_matrix,N,t_m,d_prior,beta_values):
    #assumption same number of actions that for states


    iterations=10
    utilities_action=[]
    utilities_list=[]
    utilities_optimal=[]
    utilities_list_optimal=[]
    costs=[]
    low_confidence_interval=[]
    high_confidence_interval=[]
    orizon = 5




    for i in k_samples:
       
        action_list=[]

        for j in range(iterations):
            o=1
            a=np.random.choice(states,1, p=d_prior)
            
            state=np.random.choice(states,1, p=t_m[a[0]])


            #optimal agent action
            optimal_agent = np.argmax(np.average(utility_matrix,axis=1, weights=p_true))
           
            utilities_optimal.append(np.amax(utility_matrix[state], axis=1))
             
            rejection_sampling(utility_matrix,a,state,beta_values[i],d_prior,states,
                        action_list,utilities_action,t_m,costs,orizon,o,p_true)
            

            
        #counting the number of times a certain action was chosen as argmax given the iterations


        #average utility given k observations for bounded approximate agent
        mean_utility_sampleagent= np.mean(utilities_action)

        #average utility given k observations for optimal agent
        mean_utility_optimal= np.mean(utilities_optimal)
      

        #collect utilities for k observation
        utilities_list.append(mean_utility_sampleagent)
        utilities_list_optimal.append(mean_utility_optimal)
        
        boot = bootstrap( list(itertools.chain(*utilities_action)), n=1000)
        interval=list(boot(.95))
        low_confidence_interval.append(interval[0])
        high_confidence_interval.append(interval[1])





    #Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Performance in Utility")
    ax.set_ylabel("Mean Utility ")
    ax.set_xlabel("Beta")
    x_axis = k_samples
    plt.plot(x_axis, utilities_list, 'b')
    plt.fill_between(x_axis,low_confidence_interval, high_confidence_interval,
                         color='gray', alpha=0.2)
    plt.plot(x_axis, utilities_list_optimal, 'r')
    plt.gca().legend(('Sampling Agent','Optimal Agent'))


    plt.savefig('myfig')
    plt.show()
    
    return 


def main():
	# number of samples used to for a belief about possible decisons
    k=[i for i in range(50)]
    #action_state _space n x n matrix
    n_action_state = [1000]

    #rationality of the agent
    beta_values = [i for i in range(50)]
    
    
    ###samples for forming true distribution
    N=10000

    #### alpha an beta hyperparameters for betabinomial
    a= 5
    b=5
    
    for n_as in range(len(n_action_state)) :
        
        ### vector of parameters for dirichlet distribution, FLAT PRIOR
        #number of times k samples are collected
        d_prior=normalization(sample_Beta_Binomial(1,1 , n_action_state[n_as ]-1, N),n_action_state[n_as ],N)
        print("The uniform prior is:",d_prior)

        states  = [i for i in range(n_action_state[n_as])]
        ###Sampling from a betabinomial distrubtion with given parameters(n,alpha,beta) 
        ##forming the real underling vector of parameters and this case also the generative 
        #probability distribution
        p_true_betabino=sample_Beta_Binomial(a,b ,n_action_state[n_as]-1, N)
        true_p=normalization(p_true_betabino,n_action_state[n_as],N)
      
       

        #Utility Matrix
        sparse_matrix=sp.sparse.random(n_action_state[n_as],n_action_state[n_as],density=0.7)
        sparse_matrix=sparse_matrix.A
        print(sparse_matrix)
        

        ###Defining a Transition matrix where an action is defined as a map
        # form a state into another state
        transitions=np.random.choice(states,100000, p=true_p)
        t_m=transition_matrix(transitions,n_action_state[n_as])
        
        
        
        sparse_reward= actions(n_action_state[n_as],states,k,true_p,
                                     sparse_matrix,N,t_m,true_p,beta_values)
        
        
            
   


if __name__ == '__main__':
	main()
