'''
Created on Dec 11, 2011

@author: kykamath
'''
from library.file_io import FileIO
from settings import hashtagsWithoutEndingWindowFile
from experiments.mr_area_analysis import getOccuranesInHighestActiveRegion,\
    getOccurranceDistributionInEpochs, HashtagsClassifier, getRadius
from operator import itemgetter
import math
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.svm import SVC, LinearSVC
from experiments.models import Hashtag
from sklearn.externals import joblib


timeRange, folderType = (2,11), 'world'

#class Classifier:
#    def __init__(self):
        
i = 1
documents = []
for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(folderType,'%s_%s'%timeRange)):
#    occurranceDistributionInEpochs = getOccurranceDistributionInEpochs(getOccuranesInHighestActiveRegion(h), timeUnit=CLASSIFIER_TIME_UNIT_IN_SECONDS, fillInGaps=True)
#    occuranceVector = zip(*sorted(occurranceDistributionInEpochs.iteritems(), key=itemgetter(0)))[1]
#    if h['h']=='rocklake':
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
trainX, trainy = zip(*trainDocuments)
testX, testy = zip(*testDocuments)
#clf = SVC(probability=True)
#clf.fit(trainX, trainy)
FileIO.createDirectoryForFile('classifiers/abc.pkl')
#joblib.dump(clf, 'classifiers/abc.pkl')
clf = joblib.load('classifiers/abc.pkl')
print clf.score(testX, testy)

