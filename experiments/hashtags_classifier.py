'''
Created on Dec 11, 2011

@author: kykamath
'''
import sys
sys.path.append('../')
from library.file_io import FileIO
from settings import hashtagsClassifiersFolder, hashtagsFile,\
    hashtagsAnalysisFolder
from experiments.mr_area_analysis import HashtagsClassifier
from sklearn.svm import SVC
from experiments.models import Hashtag
from sklearn.externals import joblib
from library.classes import GeneralMethods


class Classifier:
    FEATURES_RADIUS = 'radius'
    FEATURES_OCCURANCES_RADIUS = 'occurances_radius'
    FEATURES_AGGGREGATED_OCCURANCES_RADIUS = 'aggregate_occurances_radius'
    classifiersPerformanceFile = hashtagsAnalysisFolder+'/classifiers/classifier_performance'
    def __init__(self, numberOfTimeUnits, features):
        self.clf = None
        self.numberOfTimeUnits = numberOfTimeUnits
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
    def predict(self, document):
        if self.clf==None: self.clf = joblib.load(self.classfierFile)
        return self.clf.predict(document)
    def buildClassifier(self):
        documents = self._getDocuments()
        trainDocuments = documents[:int(len(documents)*0.80)]
        self.build(trainDocuments)
    def _getDocuments(self):
        documents = []
        for i, h in enumerate(FileIO.iterateJsonFromFile(hashtagsFile%('training_world','%s_%s'%(2,11)))):
            ov = Hashtag(h, dataStructuresToBuildClassifier=True)
#            print i
            if ov.isValidObject() and ov.classifiable: 
                if self.features == Classifier.FEATURES_RADIUS: documents.append(ov.getVector(self.numberOfTimeUnits, radiusOnly=True))
                elif self.features == Classifier.FEATURES_OCCURANCES_RADIUS: documents.append(ov.getVector(self.numberOfTimeUnits, radiusOnly=False))
                elif self.features == Classifier.FEATURES_AGGGREGATED_OCCURANCES_RADIUS: 
                    vector = ov.getVector(self.numberOfTimeUnits, radiusOnly=True, aggregate=True)
                    if vector[0][-1]>=HashtagsClassifier.RADIUS_LIMIT_FOR_LOCAL_HASHTAG_IN_MILES: vector[0] = [1]
                    else: vector[0] = [0]
                    documents.append(vector)
        return documents
    def testClassifierPerformance(self):
        documents = self._getDocuments()
        testDocuments = documents[-int(len(documents)*0.20):]
        print {'features': self.features, 'numberOfTimeUnits': self.numberOfTimeUnits, 'score': self.score(testDocuments)}
        FileIO.writeToFileAsJson({'features': self.features, 'numberOfTimeUnits': self.numberOfTimeUnits, 'score': self.score(testDocuments)}, Classifier.classifiersPerformanceFile)
    @staticmethod
    def testClassifierPerformances():
#        GeneralMethods.runCommand('rm -rf %s'%Classifier.classifiersPerformanceFile)
        for numberOfTimeUnits in range(1,25):
            for feature in [Classifier.FEATURES_AGGGREGATED_OCCURANCES_RADIUS, Classifier.FEATURES_OCCURANCES_RADIUS, Classifier.FEATURES_RADIUS]:
                classifier = Classifier(numberOfTimeUnits, features=feature)
                classifier.testClassifierPerformance()
    @staticmethod
    def buildClassifiers():
        for feature in [Classifier.FEATURES_AGGGREGATED_OCCURANCES_RADIUS]:
            for numberOfTimeUnits in range(1,25):
                classifier = Classifier(numberOfTimeUnits, features=feature)
                classifier.buildClassifier()

Classifier.testClassifierPerformances()