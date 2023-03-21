#MODULES

#Utility
import numpy as np
from tqdm import tqdm
import math

#Matrix Utility
from scipy.sparse import coo_matrix 
from scipy.sparse.linalg import svds  

#Plotting
import matplotlib.pyplot as plt

#Drive 
from colab import drive
drive.mount('/content/drive')

#custom functions
from dataset_to_dict import dict_from_lines
from dataset_to_dict import dict_to_ID_list
from utility import gamma_list
from utility import compute_c_vector
from utility import matrix_dot_transposed_self
from utility import cosine_matrix_gen
from training import train_gd
from training import train_sgd

#----------------------------------------------------------------------------------

#TASK 1: LOADING THE DATASET

list1 = []
#NB change path to your location
with open('/content/drive/MyDrive/Databases and data Science/netflix_challenge/combined_data_1.txt', encoding='utf8') as f:
    for line in f:
        list1.append(line)

dictionary = dict_from_lines(list1)

User_ID_list, Movie_ID_list, ratings, users, movies = dict_to_ID_list(dictionary)

n_users = User_ID_list[-1]
n_movies  = Movie_ID_list[-1]
print("movies: ", n_movies, "     users: ", n_users )
print("entries in the sparse matrix: ", len(users))


#sparse matrix generation
sparse_wide = coo_matrix((ratings, (users, movies)), shape=(n_movies + 1 , n_users + 1), dtype=np.float32)
sparse_tall = coo_matrix((ratings, (movies, users)), shape=(n_users + 1 , n_movies + 1))

#----------------------------------------------------------------------------------

#TASK 2: DIMSUM

eps_v = np.linspace(2,6,20)
gammas = gamma_list(n_movies, eps_v)

c_vector = compute_c_vector(n_movies, sparse_tall) #to speed up we compute c vectors beforehand
A_tA = matrix_dot_transposed_self(n_movies, sparse_tall) 


results = []

for gamma in tqdm(gammas):
  
  B, D = cosine_matrix_gen(n_movies, sparse_tall, c_vector, gamma )
  #print("Matrix of cosine similarities \n", B)
  #print("Diagonal Matrix \n", D)
  DBD = np.dot(D, np.dot(B,D))

  diff = (DBD - A_tA)
  result = np.linalg.norm(diff)  /  np.linalg.norm(A_tA)  #theorem from article "Dimension independent matrix square using mapreduce"
  #print(np.round(result,2))
  results.append(np.round(result,2) )

#df <- data.frame(gamma_v = gammas, eps_v = eps_v, results = results)
plt.plot(gammas,  eps_v, label = "Epsilon Values")
plt.plot(gammas,  results, label = "results")
plt.xlabel('Gamma Values')
plt.xscale("log")
plt.title('Comparison epsilon vs computed difference')
plt.legend()
plt.show()

#----------------------------------------------------------------------------------

#TASK 3: GRADIENT WITH LATENT FACTORS

matrix_for_rank = sparse_wide.todense()
rank = np.linalg.matrix_rank(matrix_for_rank)  #NB number of singular values for svd is rank of sparse matrix
matrix_for_rank = None  #free memory after the computation
print("Rank of the sparse matrix: ", rank)

sparse_wide = coo_matrix((ratings, (users, movies)), shape=(n_movies + 1 , n_users + 1), dtype=np.float32)
u, s, vT = svds(sparse_wide.tocsc(), k=rank) #compute singular value decomposition
s = np.diag(s)
# u   undarray, shape=(M, k)  Unitary matrix having left singular vectors as columns.
# s   ndarray,  shape=(k,) The singular values.
# vT  ndarray,  shape=(k, N) Unitary matrix having right singular vectors as rows.
Q = u
pT = np.dot(s,vT)
sparse_wide = sparse_wide.tocsr()
print(sparse_wide.shape)

#Gradient Descent matrix upgrade 

#-------------------PARAMETERS------------------------------------------------------
lambda1 = 0.05   #check in gradient_descent.py
lambda2 = 0.05   #check in gradient_descent.py

#Batch GD learning rate
alpha_b = 0.01

#Stochastic GD learning rate
alpha_s = 0.0000001

# Q, pT = batch_gradient_descent(sparse_wide, Q, pT, alpha_b)
#Q2, pt2 = Stochastic_GD(sparse_wide, Q, pT, alpha_s)

epochs_gd = 5      #each epoch is computationally very expensive
epochs_sgd = 20


#-------------------TRAINIG--------------------------------------------------------
gd_trained_Q, gd_trained_pT, errors_gd = train_gd(epochs_gd, Q, pT, sparse_wide, alpha_b)


sgd_trained_Q, sgd_trained_pT, errors_sgd = train_sgd(epochs_sgd, Q, pT, sparse_wide, alpha_s)


#with Optimized P and Q produced right above the matrix M = Q*P.transposed will contain predicted ratings 
#for every movie and every user. (don't compute it! Lot of memory required)
#multiply lines and rows to compute singular entries of M instead to compare it with the original sparse matrixin TASK1


#-------------------PLOTTTING------------------------------------------------------
 
#these two are the plot of the training errors through epochs
plt.plot(range(100), epochs_gd, label = "RMSE")
plt.xlabel('epochs')
plt.title('GDepochs vs accuracy')
plt.legend()
plt.show() 


plt.plot(range(100), epochs_sgd, label = "RMSE")
plt.xlabel('epochs')
plt.title('SGD epochs vs accuracy')
plt.legend()
plt.show() 


#-------------- Testing Accuracy with Training and Test Set--------------------------

flag = False  #set True if you want to compute accuracy test
if flag:

    print("Splitting in Training and Test dataset")
    #2649427 is the length of the dataset
    w = sparse_wide.tocsc()
    trainingSet = w[:,  0:2000000]
    testSet = w[:,  2000001:2649427]
    print("Training: ",trainingSet.shape," Test: ",testSet.shape)

    matrix_for_rank = trainingSet.todense()
    rank = np.linalg.matrix_rank(matrix_for_rank)
    matrix_for_rank = None

    u, s, vT = svds(trainingSet.tocsc(), k=rank) 
    s = np.diag(s)
    Q = u
    pT = np.dot(s,vT)

    #to_Test_Q, to_test_pT = train_gd(epochs_gd, Q, pT, sparse_wide, alpha_b)
    Q, pT = train_sgd(epochs_sgd, Q, pT, sparse_wide, alpha_s)

    n = np.dot(Q, pT)[:,0:649426]
    n = np.array(n)
    print("to test matrix shape ", n.shape)
    testSet = testSet.todense()
    print("test Set ", testSet.shape)

    test_error = testSet - n
    rmse_test = math.sqrt(np.sum(pow(np.array(test_error),2).flatten()))
    print("Test Root Mean-Squared error: ", rmse_test)
