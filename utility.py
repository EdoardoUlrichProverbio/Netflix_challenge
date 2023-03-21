#MODULES

#Utility
import numpy as np
import random 

#----------------------------------------------------------------------------------

def gamma_list(n_movies, eps_list):
  gamma = []
  for eps in eps_list:
    x = n_movies/(eps*eps)
    gamma.append(np.sqrt(x * np.exp(x) ) ) #Remember to check how different values of gamma affect computation
    #print(gamma)

  return gamma

#----------------------------------------------------------------------------------

def compute_c_vector(n_movies, sparse_tall):

  c_vector = [0] #sparse matrix indexes starts with 1
  for n in range(1,n_movies+1,1 ):
    q = sparse_tall.getcol(n)
    c_vector.append(np.sqrt(q.multiply(q).sum(0)))

  return c_vector

#----------------------------------------------------------------------------------

def cosine_matrix_gen(n, sparse, c_vector, gamma ):
  
  B = np.zeros((n, n))
  D = np.zeros((n, n))

  for j in range(1, n+1,1): 
    q = sparse.getcol(j)
    cj = c_vector[j]

    for i in range(1,n+1,1):
      k = sparse.getcol(i)
      ck = c_vector[i]
      condition = gamma/ (ck*cj)

      if random.uniform(0,1) < min(1, condition):
        c = q.multiply(k).sum(0)

        if condition > 1 : cosine_result = c / (ck * cj)
        if condition < 1 : cosine_result = c /gamma

        B[j-1][i-1] = cosine_result

      D[i-1][i-1] = ck

  return B, D

#----------------------------------------------------------------------------------

def matrix_dot_transposed_self(n, matrix):
  A_tA =  np.zeros((n, n))
  for j in range(1, n+1,1): 
    for i in range(1, n+1,1): 
      A_tA[j-1][i-1] = matrix.getcol(j).transpose().dot(matrix.getcol(i)).sum()

  return A_tA
