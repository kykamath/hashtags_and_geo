'''
Created on Dec 11, 2011

@author: kykamath
'''
import sys
sys.path.append('../')
from library.file_io import FileIO
from settings import hashtagsWithoutEndingWindowFile, hashtagsClassifiersFolder, hashtagsFile,\
    hashtagsAnalysisFolder
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
    FEATURES_RADIUS = 'radius'
    FEATURES_OCCURANCES_RADIUS = 'occurances_radius'
    classifiersPerformanceFile = hashtagsAnalysisFolder+'/classifiers/classifier_performance'
    def __init__(self, numberOfTimeUnits, features):
        self.clf = None
        self.features = features
        self.classfierFile = hashtagsClassifiersFolder%(self.features, numberOfTimeUnits)+'model.pkl'
    def build(self, documents):
        X, y = zip(*documents)
        self.clf = SVC(probability=True)
        self.clf.fit(X, y)
        GeneralMethods.runCommand('rm -rf %s*'%self.classfierFile)
        FileIO.createDirectoryForFile(self.classfierFile)
        joblib.dump(self.clf, self.classfierFile)
    def score(self, documents):
        testX, testy = zip(*documents)
        self.clf = self.load()
        return self.clf.score(testX, testy)
    def load(self): 
        if self.clf==None: self.clf = joblib.load(self.classfierFile)
        return self.clf
    @staticmethod
    def testClassifierPerformance():
        GeneralMethods.runCommand('rm -rf %s*'%Classifier.classifiersPerformanceFile)
        for feature in [Classifier.FEATURES_RADIUS, Classifier.FEATURES_OCCURANCES_RADIUS]:
            for numberOfTimeUnits in range(1,25):
                documents = []
                classifier = Classifier(numberOfTimeUnits, features=feature)
                for h in FileIO.iterateJsonFromFile(hashtagsFile%('training_world','%s_%s'%(2,11))):
                    ov = Hashtag(h, dataStructuresToBuildClassifier=True)
                    if ov.isValidObject() and ov.classifiable: 
                        if classifier.features == Classifier.FEATURES_RADIUS: documents.append(ov.getVector(numberOfTimeUnits, radiusOnly=True))
                        else: documents.append(ov.getVector(numberOfTimeUnits, radiusOnly=False))
                testDocuments = documents[-int(len(documents)*0.20):]
                FileIO.writeToFileAsJson({'features': classifier.features, 'numberOfTimeUnits': numberOfTimeUnits, 'score': classifier.score(testDocuments)}, Classifier.classifiersPerformanceFile)
    @staticmethod
    def buildClassifier():
        for numberOfTimeUnits in range(1,25):
            documents = []
            classifier = Classifier(numberOfTimeUnits, features=Classifier.FEATURES_RADIUS)
            for h in FileIO.iterateJsonFromFile(hashtagsFile%('training_world','%s_%s'%(2,11))):
                ov = Hashtag(h, dataStructuresToBuildClassifier=True)
                if ov.isValidObject() and ov.classifiable: 
                    if classifier.features == Classifier.FEATURES_RADIUS: documents.append(ov.getVector(numberOfTimeUnits, radiusOnly=True))
                    else: documents.append(ov.getVector(numberOfTimeUnits, radiusOnly=False))
            trainDocuments = documents[:int(len(documents)*0.80)]
            classifier.build(trainDocuments)
#        print classifier.score(testDocuments)
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

#trainDocuments = documents[:int(len(documents)*0.80)]
#testDocuments = documents[:int(len(documents)*0.20)]

#trainX, trainy = zip(*trainDocuments)
#testX, testy = zip(*testDocuments)
##clf = SVC(probability=True)
##clf.fit(trainX, trainy)
#FileIO.createDirectoryForFile('classifiers/abc.pkl')
##joblib.dump(clf, 'classifiers/abc.pkl')
#clf = joblib.load('classifiers/abc.pkl')
#print clf.score(testX, testy)

#0.937282229965


#Classifier(5).build(trainDocuments)
#print Classifier(5).score(testDocuments)
Classifier.testClassifierPerformance()