# DatasetFolder

Here all preprocessing of the Datasets can be found.

Additionally this folder provides Picklefiles for:

1. digitDataset.pickle
2. textDataset.pickle
3. textDatasetMultilcass.pickle

### digitDataset

Consists of 2000 samples with 2 classes
Class 0 is Digits between 0 and 4
Class 1 is Digits between 5 and 9

For every Class there are 1000 samples in the Dataset
#### Code
with open("../Dataset/digitDataset.pickle", "rb") as fp:
    digits = pickle.load(fp,encoding='bytes')

### textDataset

The textDataset consits of 2 different classes 0, 1, from the 20newsgroups dataset, namely the mac and windows subset. 

#### Code
with open("../Dataset/textDataset.pickle", "rb") as fp:
    text = pickle.load(fp, encoding="latin-1")


### textDatasetMulticlass

The textDatasetMulticlass consits of 20 different classes 0..19 which is the entire 20newsgroup dataset.
