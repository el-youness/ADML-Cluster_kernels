from sklearn import svm
import pickle
import random
import math


# Load dataset
with open("../Dataset/digitDataset.pickle", "rb") as fp:
	digits = pickle.load(fp,encoding='bytes')
# list of testerror for every run
acclist = []

# compute 100 tuns
for z in range(0,100):
    # tale 8 samples from class 1 and 2 for fitting process
    trainSamples = []
    trainSamples += random.sample(digits[0:1000],8)
    trainSamples += random.sample(digits[1001:],8)
    trainX = []
    trainY = []
    for x in trainSamples:
        trainX.append(x[0])
        trainY.append(x[1])
    clf = svm.SVC(gamma='scale')
    clf.fit(trainX,trainY)

    # calulate testerror
    corrects = 0.0
    count = 0.0
    for x in digits:
        pred = clf.predict(x[0].reshape(1,-1))
        count+=1.0
        if(pred == x[1]):
            corrects += 1.0
    acclist.append((count-corrects)/count)
print("mean: ")
mean = sum(acclist)/100.0
print(mean)
print("standard deviation: ")
der = 0
sq = []
for x in acclist:
    sq.append((x-mean)**2)
sqmean = sum(sq)/100.0
print(math.sqrt(sqmean))


# Text Classification Test
with open("../Dataset/textDataset.pickle", "rb") as fp:
    text = pickle.load(fp, encoding="latin-1")
class1 = []
class2 = []
for s in text:
    if(s[1] == 0):
        class1.append(s)
    else:
        class2.append(s)
numbers = [2,4,8,16,32,64,128]
#print(len(class1)) 963
#print(len(class2)) 988
test = []
random.shuffle(class1)
random.shuffle(class2)
train1 = class1[:481]
train2 = class2[:494]
class1 = class1[481:]
class2 = class2[494:]
test += class1
test += class2
accs = []
trainacc = []
for n in numbers:
    l = int(n / 2)

    acclist = []
    for i in range(100):
        # train svm
        trainSample = []
        trainX = []
        trainY = []
        trainSample += random.sample(train1,l)
        trainSample += random.sample(train2,l)
        for x in trainSample:
            X = x[0].toarray()
            trainX.append(X[0])
            trainY.append(x[1])
        clf = svm.SVC(gamma='scale')
        clf.fit(trainX,trainY)

        # predict
        corrects = 0.0
        count = 0.0
        for x in test:
            pred = clf.predict(x[0].toarray()[0].reshape(1,-1))
            count+=1.0
            if(pred == x[1]):
                corrects += 1.0
        acclist.append((count-corrects)/count)
    trainacc.append(acclist)
for acc in trainacc:
    accs.append((sum(acc) / len(acc)))
print("Printing Mean Testerror for 2, 4, 8, 16, 32, 64, 128")
for acc in accs:
    print(acc)
sqList = []
for l in trainacc:
    sq = []
    mean = sum(s)/len(l)
    for x in l:
        sq.append((x-mean)**2)
    sqList.append(math.sqrt(sum(sq)/len(sq)))
print("printing derivation")
for s in sqList:
    print(s)
