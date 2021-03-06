'''
Created on Jan 14, 2012

@author: kykamath
'''
import sys
sys.path.append('../')
from settings import hashtagsFile,\
    hashtagsLatticeGraphFile
from library.file_io import FileIO
from experiments.models import Hashtag,\
    TargetSelectionRegressionSVMRBFClassifier,\
    TargetSelectionRegressionClassifier,\
    TargetSelectionRegressionSVMPolyClassifier,\
    TargetSelectionRegressionSVMLinearClassifier
from experiments.mr_area_analysis import HashtagsClassifier
from itertools import groupby
from operator import itemgetter
from collections import defaultdict
from library.classes import GeneralMethods
import numpy as np

def build(numberOfTimeUnits=24):
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
    documents = [(d, TargetSelectionRegressionClassifier.getPercentageDistributionInLattice(d)) for d in documents]
    documents = documents[:int(len(documents)*0.80)]
    for decisionTimeUnit in range(1, numberOfTimeUnits+1):
        for latticeCount, predictingLattice in enumerate(lattices):
            print decisionTimeUnit, latticeCount,
            inputVectors, outputValues = [], []
            for rawDocument, processedDocument in documents:
                documentForTimeUnit = TargetSelectionRegressionClassifier.getPercentageDistributionInLattice(rawDocument[:decisionTimeUnit])
                if documentForTimeUnit and processedDocument:
                    vector =  [documentForTimeUnit.get(l, 0) for l in lattices]
                    inputVectors.append(vector), outputValues.append(float(processedDocument.get(predictingLattice, 0)))
#            TargetSelectionRegressionClassifier(decisionTimeUnit=decisionTimeUnit, predictingLattice=predictingLattice).build(zip(inputVectors, outputValues))
            TargetSelectionRegressionSVMRBFClassifier(decisionTimeUnit=decisionTimeUnit, predictingLattice=predictingLattice).build(zip(inputVectors, outputValues))
#            TargetSelectionRegressionSVMLinearClassifier(decisionTimeUnit=decisionTimeUnit, predictingLattice=predictingLattice).build(zip(inputVectors, outputValues))
#            TargetSelectionRegressionSVMPolyClassifier(decisionTimeUnit=decisionTimeUnit, predictingLattice=predictingLattice).build(zip(inputVectors, outputValues))
#            for iv, ov in zip(inputVectors, outputValues):
#                print ov, TargetSelectionRegressionClassifier(decisionTimeUnit=decisionTimeUnit, predictingLattice=predictingLattice).predict(iv)

def testClassifierPerformance(numberOfTimeUnits=24):
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
    documents = [(d, TargetSelectionRegressionClassifier.getPercentageDistributionInLattice(d)) for d in documents]
    documents = documents[-int(len(documents)*0.20):]
    GeneralMethods.runCommand('rm -rf %s'%TargetSelectionRegressionClassifier.classifiersPerformanceFile)
    for decisionTimeUnit in range(1, numberOfTimeUnits+1):
        for classifierType in [TargetSelectionRegressionSVMRBFClassifier, TargetSelectionRegressionSVMLinearClassifier,
                               TargetSelectionRegressionSVMPolyClassifier, TargetSelectionRegressionClassifier]:
            totalError = []
            for latticeCount, predictingLattice in enumerate(lattices):
                inputVectors, outputValues, tempError = [], [], []
                for rawDocument, processedDocument in documents:
                    documentForTimeUnit = TargetSelectionRegressionClassifier.getPercentageDistributionInLattice(rawDocument[:decisionTimeUnit])
                    if documentForTimeUnit and processedDocument:
                        vector =  [documentForTimeUnit.get(l, 0) for l in lattices]
                        inputVectors.append(vector), outputValues.append(float(processedDocument.get(predictingLattice, 0)))
                classifier = classifierType(decisionTimeUnit=decisionTimeUnit, predictingLattice=predictingLattice)
                for iv, ov in zip(inputVectors, outputValues): 
                    if latticeCount==2: print ov, classifier.predict(iv), pow(ov-classifier.predict(iv), 2)
                    if ov!=0.0: 
                        tempError.append(pow(ov-classifier.predict(iv), 2))
#                print tempError, np.mean(tempError)
#                exit()
                totalError.append(np.mean(tempError))
            print {'id': classifier.id, 'timeUnit': decisionTimeUnit-1, 'error': np.mean(totalError)}
            FileIO.writeToFileAsJson({'id': classifier.id, 'timeUnit': decisionTimeUnit-1, 'error': np.mean(totalError)}, TargetSelectionRegressionClassifier.classifiersPerformanceFile)
if __name__ == '__main__':
#    build()
    testClassifierPerformance()
#    print loadLattices()
#    print len(loadLattices())
#    for data in FileIO.iterateJsonFromFile(hashtagsLatticeGraphFile%('world','%s_%s'%(2,11))):
#        print data['id']