#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Amelie
"""

import pickle 
import numpy as np 
from numpy.linalg import matrix_power
from numpy.core import multiarray
import pandas as pd
from scipy.spatial.distance import minkowski
from scipy.spatial.distance import euclidean
from numpy.linalg import matrix_power

"""prepare data"""

with open('textDataset.pickle', 'rb') as fp:
    text = pickle.load(fp, encoding="latin-1")
    
# Separate Targets from features
Xs = []
labels = []
print("textdataset size before filtering ",len(text))
for i in range(0, len(text)):
    #THRESHOLD => Check if the dimensionality of the data is inf to 360 to filter out some words.
    if(len(text[i][0].data) < 360):
        Xs.append(text[i][0].data)
        labels.append(text[i][1])
  
#len Xs = 1919       
print("new textDatasetsize ",len(Xs))

#extend NDARRAYS length
def extend(list_to_extend, target_len):
    '''
    a = np.ndarray((2,), buffer=np.array([1,2,3])
    >>> array([2, 3])
    extend(a, 5)
    >>> array([2, 3, 0, 0, 0])
    '''
    return np.append(list_to_extend,np.zeros((target_len-len(list_to_extend),)))

def max_len_of_lists(lists):
    maxLen = len(lists[0])
    for i in range(1, len(lists)):
        listLen = len(lists[i])
        if listLen > maxLen:
            maxLen = listLen
    return maxLen  

maxLen = max_len_of_lists(Xs)
for i in range(0, len(labels)):
    Xs[i] = extend(Xs[i],maxLen)
    
###########################################################   
def dist(x, y, sigma=1):
    exponent = - ((euclidean(x, y))) / (2*(sigma)**2)
    return np.exp(exponent)   
    
def compute_w(X, sigma=1):
    nb_samples = len(X)
    W = np.zeros((nb_samples, nb_samples))
    for i in range(len(X)):
        for j in range(len(X)):
            W[i, j] = dist(X[i], X[j],sigma)
    return W

def compute_a(W):
    nb_samples = len(W)
    A = np.zeros((nb_samples,nb_samples))
    for i in range (nb_samples) : 
        for k in range (nb_samples) : 
            A[i,k]=W[i,k]/sum((W.T)[i])
    return A
            
def compute_p(A,t=8):
    P = matrix_power(A,t)
    return P

def em_algo(P,Labels,nb_it=50):
    N = len(P)
    L = len(Labels)
    proba_label = 0.5 * np.ones((N,2))
    proba_i =  1/N *np.ones((N,L))
    for it in range(nb_it):
    
        for i in range (N) :
            for k in range (L):
                proba_i[i,k]=proba_label[i,Labels[k]]*P[i,k]
                
        #M-step 
        for i in range(N):
            for y in range(2):
                num = 0 
                denom = 0
                for k in range(L):
                    if Labels[k]==y:
                        num = num+proba_i[i,k]
                    denom = denom + proba_i[i,k]
                proba_label[i,y]=num/denom
                
    return proba_label

def compute_result(proba_label):
    result = []
    nb_samples = len(proba_label)
    for i in range (nb_samples):
        sample = np.argmax(proba_label[i])
        result.append(sample)
    return result 

def train_error(labels,result):
    nb = len(result)
    error = 0 
    for i in range(nb):
        error = error + abs(labels[i]-result[i])
    return error / nb    
######################################################
    
def random_walk(X,labels,sig,t_here,nb_labelled_point,em_iterations,nb_points):
    #nb_samples = len(Xs)     
    #dim = len(Xs[0])   
 
    labels_eff = labels[0+400:nb_labelled_point+400]
    new_Xs = Xs[0:nb_points]
    new_labels = labels[0:nb_points]

    W = compute_w(new_Xs,sig)
    A = compute_a(W)
    P = compute_p(A,t_here)
    proba_label = em_algo(P,labels_eff,em_iterations)
    result = compute_result(proba_label)
    error = train_error(new_labels,result)  
    
    return error 

error = random_walk(Xs,labels,0.6,8,120,200,1919)
