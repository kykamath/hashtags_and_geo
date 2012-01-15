'''
Created on Jan 14, 2012

@author: kykamath
'''
from settings import targetSelectionRegressionClassifiersFolder, hashtagsFile,\
    hashtagsLatticeGraphFile
from library.file_io import FileIO
from experiments.models import Hashtag
from experiments.mr_area_analysis import HashtagsClassifier
from itertools import groupby
from operator import itemgetter
from collections import defaultdict

class TargetSelectionRegressionClassifier:
    def __init__(self, id='linear_regression'): self.id = id
    def getModelFile(self, decisionTimeUnit, predictingLattice):
        modelFile = targetSelectionRegressionClassifiersFolder%(self.id, decisionTimeUnit, predictingLattice)+'model.pkl'
        FileIO.createDirectoryForFile(modelFile)
        return modelFile
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
    documents = [(d, getPercentageDistributionInLattice(d)) for d in documents]
    for timeUnit in range(1, numberOfTimeUnits+1):
        for lattice in lattices:
            inputVectors, outputValues = [], []
            for rawDocument, processedDocument in documents:
                documentForTimeUnit = getPercentageDistributionInLattice(rawDocument[:timeUnit])
                if documentForTimeUnit and processedDocument:
                    vector =  [documentForTimeUnit.get(l, 0) for l in lattices]
                    inputVectors.append(vector), outputValues.append(float(processedDocument.get(lattice, 0)))
            from sklearn import linear_model
            clf = linear_model.LinearRegression()
            clf.fit (inputVectors, outputValues)
            print outputValues[0], clf.predict(inputVectors[0])
            exit()
#        print documents
#        exit()
if __name__ == '__main__':
    build()
#    for data in FileIO.iterateJsonFromFile(hashtagsLatticeGraphFile%('world','%s_%s'%(2,11))):
#        print data['id']