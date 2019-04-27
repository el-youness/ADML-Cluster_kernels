import numpy as np
import sklearn.utils.extmath as sm
from numpy.linalg import inv
from numpy.linalg import eig
from numpy import dot, diag
from scipy.linalg import sqrtm
from scipy.spatial.distance import euclidean
import random, math
np.random.seed(42)
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def fill_diag(M, a):
    """
    M: square matrix
    a: array of length number of rows
    ----
    fill the diagonal of M with values of array a
    """
    s = M.shape
    D = np.zeros(s)
    for i in range(s[0]):
        D[i,i] = a[i]
    return D

def rbf_kernel(X, sigma=1):
    K = np.zeros((len(X), len(X)))
    for a in range(len(X)):
        for b in range(len(X)):
            K[a, b] = rbf_function(X[a], X[b],sigma)
    return K
            
def rbf_function(x, y, sigma=1):
    exponent = - (euclidean(x, y) ** 2) / (2 * (sigma ** 2))
    return np.exp(exponent)


def diagonal_row_sum_matrix(M):
    rows_sum = M.sum(axis = 1)
    return fill_diag(M,rows_sum)

def computeL(D,K):
    Dinv = inv(D)
    return sqrtm(Dinv).dot(K).dot(sqrtm(Dinv))

def pick_eigs(eigen_vals, eigen_vect, k):
    if k > len(eigen_vals):
        k = len(eigen_vals)
    eig_vals = list(eigen_vals)
    new_eigs = []
    new_eigen_vector = []
    last_max_value = 0
    for i in range(0, k):
        argmax = eig_vals.index(max(eig_vals))
        new_eigen_vector.append(eigen_vect[argmax])
        new_eig = eig_vals.pop(argmax)
        new_eigs.append(new_eig)
        last_max_value = new_eig
        
    argmax = eig_vals.index(max(eig_vals))
    new_eig = eig_vals[argmax]
        
    while(new_eig == last_max_value):
        argmax = eig_vals.index(max(eig_vals))
        new_eigen_vector.append(eigen_vect[argmax])
        new_eig = eig_vals.pop(argmax)
        new_eigs.append(new_eig)
        
    return np.array(new_eigs), np.array(new_eigen_vector)      
    

def build_K(lambdaCut, transfer, X, n_clusters, sigma=5):
    
    #Step 1 - K matrix
    K = rbf_kernel(X, sigma)
    D = diagonal_row_sum_matrix(K)
    
    #Step 2 - L matrix
    L = computeL(D, K)
    eigen_vals, U = eig(L)
    eigen_vals, U = pick_eigs(eigen_vals, U, n_clusters)   
    
    Q = diag(eigen_vals)
    
    #Step 3 - Transfer Function
    #choosing lambdacut
    newEigen = transfer(eigen_vals, lambdaCut)
    newEigen = diag(newEigen)
    
    #Step 4 - New Kernel matrix
#     print(U.shape)26
#     print(newEigen.shape)22   6222 62 26
    newL = (U.T).dot(newEigen).dot((U))
    newD = inv(diag(diag(L)))
    newK = sqrtm(newD).dot(newL).dot(sqrtm(newD))
    return newK
    

#TRANSFER FUNCTION
def linear(vals, lambdaCut):
    return vals

def step(vals,lambdaCut):
    return [ 1 if x >= lambdaCut else 0 for x in vals ]

def linear_step(vals, lambdaCut):
    return [ x if x >= lambdaCut else 0 for x in vals ]

def polynomial(vals, exponent):
    return [ np.power(x, exponent) for x in vals ]

def polystep(vals, lambdaCut):
    return [ np.power(x, 2) if x > lambdaCut else np.power(x, 2) for x in vals ]

#data load
print('DataLoad')
from sklearn import svm
import pandas
import time
dataset = pandas.read_csv("../DataSet/HTRU_2.csv")
dataset = dataset.values
labels1 = []
labels2 = []
vectors1 = []
vectors2 = []
for l in dataset:
    if(l[-1] == 0):
        labels1.append(l[-1])
        vectors1.append(l[:-1])
    else:
        labels2.append(l[-1])
        vectors2.append(l[:-1])
vectors1 = vectors1[:1000]
labels1 = labels1[:1000]
vectors = vectors1+vectors2
labels = labels1+labels2
print(len(vectors1))
print(len(vectors2))
print(type(vectors1))
print(type(labels1))
# data preprocess
print('Data Preprocess')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(vectors)
vectors1 = scaler.transform(vectors1)
vectors2 = scaler.transform(vectors2)
vectors = scaler.transform(vectors)
vectors1 = list(vectors1)
vectors2 = list(vectors2)


# SVM with K matrix as Text classification
lambdaCut_or_polyDegree = 5
print('Build K')
start = time.time()
K = build_K(lambdaCut_or_polyDegree, polynomial, vectors, 2, sigma=0.5)
end = time.time()
print('Took: ', end-start)
accs = []
testInd = list(range(len(labels)))
print('Start Modelling')
for i in range(0,10):
    print('RUN: ', i)
    trainx = []
    trainInd = []
    trainy = []
    labeledPerClass = 8
    c1 = 0
    c2 = 0
    finished = False
    while(not finished):
        index = random.randint(0,len(labels)-1)
        if(labels[index] == 0 and c1 < labeledPerClass):
            trainInd.append(index)
            trainy.append(labels[index])
            c1 += 1
        if(labels[index] == 1 and c2 < labeledPerClass):
            trainInd.append(index)
            trainy.append(labels[index])
            c2 += 1
        if(c1 == labeledPerClass and c2 == labeledPerClass):
            finished = True
    KtrainX = K[np.ix_(trainInd, trainInd)]
    #print('Finished making trainingset')
    clf = svm.SVC(gamma='scale')
    clf.fit(KtrainX, trainy)
    #print('Fitting finished')
    #print('Predicting')
    KtestX = K[np.ix_(testInd, trainInd)]
    pred = clf.predict(KtestX)
    acc = np.sum(labels == pred)/len(labels)
    print(acc)
    accs.append(acc)
print('Mean of acc = ' , np.mean(accs))
print()
print()

accs = []

for i in range(0,10):
    labeledPerClass = 8
    trainX = random.sample(vectors1,labeledPerClass) + random.sample(vectors2,labeledPerClass)
    trainY = random.sample(labels1,labeledPerClass) + random.sample(labels2,labeledPerClass)
    testX = vectors

    clf= svm.SVC(gamma='scale')
    clf.fit(trainX,trainY)

    pred = clf.predict(testX)
    acc = np.sum(labels==pred)/len(labels)
    print(acc)
    accs.append(acc)
print('Mean of acc = ' , np.mean(accs))
