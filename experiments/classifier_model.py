'''
Created on Jan 14, 2012

@author: kykamath
'''
import sys
sys.path.append('../')
from settings import targetSelectionRegressionClassifiersFolder, hashtagsFile,\
    hashtagsLatticeGraphFile
from library.file_io import FileIO
from experiments.models import Hashtag
from experiments.mr_area_analysis import HashtagsClassifier
from itertools import groupby
from operator import itemgetter
from collections import defaultdict
from sklearn import linear_model
from library.classes import GeneralMethods
from sklearn.externals import joblib

class TargetSelectionRegressionClassifier:
    def __init__(self, id='linear_regression', decisionTimeUnit=None, predictingLattice=None): 
        self.id = id
        self.decisionTimeUnit = decisionTimeUnit
        self.predictingLattice = predictingLattice
        self.classfierFile = targetSelectionRegressionClassifiersFolder%(self.id, self.decisionTimeUnit, self.predictingLattice)+'model.pkl'
        FileIO.createDirectoryForFile(self.classfierFile)
        self.clf = None
    def build(self, trainingDocuments):
        inputVectors, outputValues = zip(*trainingDocuments)
        clf = linear_model.LinearRegression()
        clf.fit (inputVectors, outputValues)
        GeneralMethods.runCommand('rm -rf %s*'%self.classfierFile)
        FileIO.createDirectoryForFile(self.classfierFile)
        joblib.dump(clf, self.classfierFile)
    def predict(self, vector):
        if self.clf==None: self.clf = joblib.load(self.classfierFile)
        return self.clf.predict(vector)
def build(numberOfTimeUnits=24):
    def getPercentageDistributionInLattice(document):
        data = zip(*document)[1]
        distributionInLaticces = defaultdict(int)
        for d in data:
            for k, v in d: distributionInLaticces[k]+=v
        total = float(sum(distributionInLaticces.values()))
        return dict([k,v/total] for k, v in distributionInLaticces.iteritems())
    validLattices = set()
    for data in FileIO.iterateJsonFromFile(hashtagsLatticeGraphFile%('world','%s_%s'%(2,11))): validLattices.add(data['id'])
    documents, lattices = [], set()
    for h in FileIO.iterateJsonFromFile(hashtagsFile%('training_world','%s_%s'%(2,11))): 
        hashtag, document = Hashtag(h), []
        if hashtag.isValidObject():
            for timeUnit, occs in enumerate(hashtag.getOccrancesEveryTimeWindowIterator(HashtagsClassifier.CLASSIFIER_TIME_UNIT_IN_SECONDS)):
                occs = filter(lambda t: t[0] in validLattices, occs)
                occs = sorted(occs, key=itemgetter(0))
                if occs: 
                    for lattice in zip(*occs)[0]: lattices.add(lattice)
                document.append([timeUnit, [(k, len(list(i))) for k, i in groupby(occs, key=itemgetter(0))]])
            if document: documents.append(document)
    lattices = sorted(list(lattices))
    print len(lattices)
    documents = [(d, getPercentageDistributionInLattice(d)) for d in documents]
    for decisionTimeUnit in range(1, numberOfTimeUnits+1):
        for latticeCount, predictingLattice in enumerate(lattices):
            print decisionTimeUnit, latticeCount,
            inputVectors, outputValues = [], []
            for rawDocument, processedDocument in documents:
                documentForTimeUnit = getPercentageDistributionInLattice(rawDocument[:decisionTimeUnit])
                if documentForTimeUnit and processedDocument:
                    vector =  [documentForTimeUnit.get(l, 0) for l in lattices]
                    inputVectors.append(vector), outputValues.append(float(processedDocument.get(predictingLattice, 0)))
#            TargetSelectionRegressionClassifier(decisionTimeUnit=decisionTimeUnit, predictingLattice=predictingLattice).build(zip(inputVectors, outputValues))
            for iv, ov in zip(inputVectors, outputValues):
                print ov, TargetSelectionRegressionClassifier(decisionTimeUnit=decisionTimeUnit, predictingLattice=predictingLattice).predict(iv)
            exit()
#        print documents
#        exit()
if __name__ == '__main__':
    build()
#    for data in FileIO.iterateJsonFromFile(hashtagsLatticeGraphFile%('world','%s_%s'%(2,11))):
#        print data['id']