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
import random 
import matplotlib.pyplot as plt 

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
    L = len(P)
    N = len(Labels)
    proba_label = 0.5 * np.ones((L,2))
    proba_i =  1/N *np.ones((L,N,2))
    for it in range(nb_it):
    
        for i in range (L) :
            for k in range (N):
                for lab in range(2):
                    proba_i[i,k,lab]=proba_label[i,lab]*P[i,k]
                
        for i in range(L):
            for y in range(2):
                num = 0 
                denom = 0
                for k in range(N):
                    if int(Labels[k])==y:
                        num = num+proba_i[i,k,y]
                    denom = denom + proba_i[i,k,int(Labels[k])]
                
                proba_label[i,y]=num/denom
                
    return proba_label

def compute_p_post(P,Labels, proba_label):
    N = len(P)
    L = len(Labels)
    P_post = 0.5 * np.ones((N,2))
    for y in range(2):
        for k in range(N):
            somme = 0
            for i in range(N):
                here = proba_label[i,y]*P[i,k]
                somme = somme + here
            P_post[k,y]=somme
    return P_post



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
 
    labels_eff = labels[0:nb_labelled_point]
    new_Xs = Xs[0:nb_points]
    new_labels = labels[0:nb_points]

    W = compute_w(new_Xs,sig)
    A = compute_a(W)
    P = compute_p(A,t_here)
    proba_label = em_algo(P,labels_eff,em_iterations)
    P_post = compute_p_post(P,labels, proba_label)
    result = compute_result(P_post)
    error = train_error(new_labels,result)  
    
    return [proba_label,P_post,result,error]  

#error = random_walk(Xs,labels,0.6,8,120,200,1919)
    
def random_walk2(X,labels,sig,t_here,em_iterations,labels_ref):
    #nb_samples = len(Xs)     
    #dim = len(Xs[0])   

    W = compute_w(X,sig)
    A = compute_a(W)
    P = compute_p(A,t_here)
    proba_label = em_algo(P,labels,em_iterations)
    P_post = compute_p_post(P,labels, proba_label)
    result = compute_result(P_post)
    error = train_error(labels_ref,result)  
    
    return [proba_label,P_post,result,error] 


##########################
# non linear data 
########################
    
n = 100
Test = np.ones((n*2,2))
Test_full = np.ones((n*2,3))
Test_full2 = np.ones((n*2,3))
test_labels = []
y1 = [((random.random()*4)+3) for i in range(n)]
x1 = []
for i in range(n):
    here = (y1[i]-5)**2+1
    x1.append(here+random.random())  
plt.plot(x1,y1,'o')

y2 = [((random.random()*4)+1) for i in range(n)]
x2 = []
for i in range(n):
    here = -((y2[i]-3)**2)+7
    x2.append(here+random.random())  
plt.plot(x2,y2,'o')

for i in range(n):
    Test[i,0]=x1[i]
    Test[i,1]=y1[i]
    test_labels.append(0)
    Test_full[i,0]=x1[i]
    Test_full[i,1]=y1[i]
    Test_full[i,2]=0
for i in range(n):
    Test[i+n,0]=x2[i]
    Test[i+n,1]=y2[i]
    test_labels.append(1)
    Test_full[i+n,0]=x2[i]
    Test_full[i+n,1]=y2[i]
    Test_full[i+n,2]=1
       
#random.seed(1)
#random.shuffle(Test_full)
    
#for i in range (n):
#    Test_full2[i]=Test_full[i*2]
#    Test_full2[i+n]=Test_full[(i*2)+1]
    

    
for i in range (int(2*n/20)):
    Test_full2[i]=Test_full[i*20]
    Test_full2[i+int((2*n/20))]=Test_full[(i*20)+1]
    Test_full2[i+int((2*n/20)*2)]=Test_full[(i*20)+2]
    Test_full2[i+int((2*n/20)*3)]=Test_full[(i*20)+3]
    Test_full2[i+int((2*n/20)*4)]=Test_full[(i*20)+4]
    Test_full2[i+int((2*n/20)*5)]=Test_full[(i*20)+5]
    Test_full2[i+int((2*n/20)*6)]=Test_full[(i*20)+6]
    Test_full2[i+int((2*n/20)*7)]=Test_full[(i*20)+7]
    Test_full2[i+int((2*n/20)*8)]=Test_full[(i*20)+8]
    Test_full2[i+int((2*n/20)*9)]=Test_full[(i*20)+9]
    Test_full2[i+int((2*n/20)*10)]=Test_full[(i*20)+10]
    Test_full2[i+int((2*n/20)*11)]=Test_full[(i*20)+11]
    Test_full2[i+int((2*n/20)*12)]=Test_full[(i*20)+12]
    Test_full2[i+int((2*n/20)*13)]=Test_full[(i*20)+13]
    Test_full2[i+int((2*n/20)*14)]=Test_full[(i*20)+14]
    Test_full2[i+int((2*n/20)*15)]=Test_full[(i*20)+15]
    Test_full2[i+int((2*n/20)*16)]=Test_full[(i*20)+16]
    Test_full2[i+int((2*n/20)*17)]=Test_full[(i*20)+17]
    Test_full2[i+int((2*n/20)*18)]=Test_full[(i*20)+18]
    Test_full2[i+int((2*n/20)*19)]=Test_full[(i*20)+19]


les_labels = list(Test_full2[:10,2])
la_ref = list(Test_full2[:,2])

[proba_label,P_post,result,error]  = random_walk2(Test_full2[:,0:2],les_labels,0.6,8,100,la_ref)
print(error)

#################################
# for text data set
###############################

#[proba_label,P_post,result,error] = random_walk(Xs,labels,1,8,100,50,100)

