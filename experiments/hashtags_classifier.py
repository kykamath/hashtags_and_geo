'''
Created on Dec 11, 2011

@author: kykamath
'''
from library.file_io import FileIO
from settings import hashtagsWithoutEndingWindowFile, hashtagsClassifiersFolder
from experiments.mr_area_analysis import getOccuranesInHighestActiveRegion,\
    getOccurranceDistributionInEpochs, HashtagsClassifier, getRadius
from operator import itemgetter
import math
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.svm import SVC, LinearSVC
from experiments.models import Hashtag
from sklearn.externals import joblib
from library.classes import GeneralMethods


timeRange, folderType = (2,11), 'world'

class Classifier:
    def __init__(self, numberOfTimeUnits, folderType='world'):
        self.classfierFile = hashtagsClassifiersFolder%(folderType, numberOfTimeUnits)+'model.pkl'
    def build(self, documents):
        X, y = zip(*documents)
        clf = SVC(probability=True)
        clf.fit(X, y)
        GeneralMethods.runCommand('rm -rf %s*'%self.classfierFile)
        FileIO.createDirectoryForFile(self.classfierFile)
        joblib.dump(clf, self.classfierFile)
    def load(self): return joblib.load(self.classfierFile)
        
i = 1
documents = []
for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(folderType,'%s_%s'%timeRange)):
    ov = Hashtag(h, dataStructuresToBuildClassifier=True)
    if ov.isValidObject() and ov.classifiable: documents.append(ov.getVector(5))
    print i
    i+=1
    if i==200: break
    

#X, y = zip(*documents)
#X = np.array(X)
#y = np.array(y)
#kf = KFold(n=len(documents), k=3)
#for i, (train_index, test_index) in enumerate(kf):
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]
#    clf = LinearSVC()
#    clf.fit(X_train, y_train)
#    print clf.score(X_test, y_test)

trainDocuments = documents[:int(len(documents)*0.80)]
testDocuments = documents[:int(len(documents)*0.20)]
#trainX, trainy = zip(*trainDocuments)
testX, testy = zip(*testDocuments)
##clf = SVC(probability=True)
##clf.fit(trainX, trainy)
#FileIO.createDirectoryForFile('classifiers/abc.pkl')
##joblib.dump(clf, 'classifiers/abc.pkl')
#clf = joblib.load('classifiers/abc.pkl')
#print clf.score(testX, testy)


#Classifier(5).build(trainDocuments)
clf = Classifier(5).load()
print clf.score(testX, testy)