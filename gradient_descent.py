#MODULES

#Utility
import numpy as np
from tqdm import tqdm
import random 

#Parameters
lambda1 = 0.05
lambda2 = 0.05

#----------------------------------------------------------------------------------

def batch_gradient_descent (sparse_matrix, Q, pT, alpha_b):

  k = np.dot(Q,pT) - sparse_matrix 
  pT = pT.transpose()

  for x in tqdm(range(pT.shape[0])):
    for i in range(Q.shape[0]):

        c = k[i,x]
        pT[x] = pT[x] - alpha_b * (c*Q[i] + lambda1*pT[x])
        Q[i]  =  Q[i] - alpha_b * (c*pT[x]+ lambda2* Q[i]) 

  pT = pT.transpose() 
  return Q, pT

#----------------------------------------------------------------------------------

def stochastic_GD (sparse_matrix, Q, pT, alpha_s):

  k = np.dot(Q,pT) - sparse_matrix 
  pT = pT.transpose()
  max_iter = 100000
  iter = 1
  #while iter < max_iter:
  for i in tqdm(range(max_iter)):

    x,i = random.randint(1, pT.shape[0]-1), random.randint(1, Q.shape[0]-1)
    c = k[i,x]
    #pT[x] -=     2 * pT[:,x] * lambda1  - alpha_s * ( c * Q[i]    )  
    pT[x] = pT[x] - alpha_s* (c*Q[i] + lambda1*pT[x])
    #Q[i]    -=     2 * Q[i]  * lambda2  - alpha_s * ( c * pT[:,x] ) 
    Q[i] =  Q[i] - alpha_s * (c*pT[x]+ lambda2*Q[i]) 
    iter =+ 1
  pT = pT.transpose() 
  return Q, pT