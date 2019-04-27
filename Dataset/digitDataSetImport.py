# Imports and preprocessed the TextCLassification Data

import h5py
import random
import pickle

""" Loads original data from file and returns data and labels seperately"""
def loadDataFromFile():
	with h5py.File("usps.h5", 'r') as hf:
		train = hf.get('train')
		dataTrain = train.get('data')[:]
		labelsTrain = train.get('target')[:]
	return dataTrain, labelsTrain



"""  

Returns one list in the form of  [[data, label], [data,label]...] 
Where the lables are either
	0 for digit smaller 5
	1 for digits equals or bigger 5
With exactly 1000 random samples from class1 and class2
(So a total of 2000 samples)
"""
def getFullSet():
	data, labels = loadDataFromFile()
	set1 = []
	set2 = []

	for i in range(0, len(labels)):
		if(labels[i] < 5):
			set1.append([data[i],0])
		else:
			set2.append([data[i],1])

	subset1 = random.sample(set1, 1000)
	subset2 = random.sample(set2, 1000)

        FullSet = subset1 + subset2
        TestSet = set1 + set2
	return FullSet, TestSet


def pickleData():
	dataset, testset = getFullSet()
	with open("digitDataset.pickle", "wb") as fp:
		pickle.dump(dataset, fp, protocol = 2)
	with open("digitTestset.pickle", "wb") as fp:
		pickle.dump(testset, fp, protocol = 2)


