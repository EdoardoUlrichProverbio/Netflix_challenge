#MODULES

#Utility
import numpy as np
from tqdm import tqdm
import math

#Custom function
from gradient_descent import batch_gradient_descent
from gradient_descent import stochastic_GD

#----------------------------------------------------------------------------------

def train_gd(epochs_gd, Q, pT, sparse_wide, alpha_b):
    
    trainQ, trainpT = Q, pT
    errors_gd = []
    for i in tqdm(range(epochs_gd)):
        trainQ, trainpT = batch_gradient_descent(sparse_wide, trainQ, trainpT, alpha_b)
        error = np.dot(trainQ, trainpT) - sparse_wide
        rmse = math.sqrt(np.sum(pow(np.array(error),2).flatten()))
        errors_gd.append(rmse)
        #print(rmse)
    return trainQ, trainpT, errors_gd



def train_sgd(epochs_sgd, Q, pT, sparse_wide, alpha_s):
    
    trainQ, trainpT = Q, pT
    errors_sgd = []
    for i in tqdm(range(epochs_sgd)):
        trainQ, trainpT = stochastic_GD(sparse_wide, trainQ, trainpT, alpha_s)
        error = np.dot(trainQ, trainpT) - sparse_wide
        rmse = math.sqrt(np.sum(pow(np.array(error),2).flatten()))
        errors_sgd.append(rmse)
        #print(rmse)
    return trainQ, trainpT, errors_sgd