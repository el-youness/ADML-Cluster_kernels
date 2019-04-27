# Imports and preprocessed the TextCLassification Data

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

"""Returns a list of the form [[data, label], [data, label]...] with the textclassification"""
""" As in the paper only 2 catogieres are used """
def getSubVectorsLabels():
	categories = ["comp.windows.x", "comp.sys.mac.hardware"]
	newsgroupsSet = fetch_20newsgroups(subset='all', categories=categories)

	vectorizer = TfidfVectorizer()
	vectors = vectorizer.fit_transform(newsgroupsSet.data)

	FullSet = []
	for i in range(0, len(newsgroupsSet.target)):
		FullSet.append([vectors[i], newsgroupsSet.target[i]])

	return FullSet


"""Returns a list of the form [[data, label], [data, label]...] with the textclassification"""
""" Returns full Dataset"""
def getVectorsLabels():
	newsgroupsSet = fetch_20newsgroups(subset='all')

	vectorizer = TfidfVectorizer()
	vectors = vectorizer.fit_transform(newsgroupsSet.data)

	FullSet = []
	for i in range(0, len(newsgroupsSet.target)):
		FullSet.append([vectors[i], newsgroupsSet.target[i]])

	return FullSet

def pickleAllData():
	dataset = getSubVectorsLabels()
	with open("textDataset.pickle", "wb") as fp:
		pickle.dump(dataset, fp, protocol = 2)
	dataset = getVectorsLabels()
	with open("textDatasetMulticlass.pickle", "wb") as fp:
		pickle.dump(dataset, fp, protocol = 2)
