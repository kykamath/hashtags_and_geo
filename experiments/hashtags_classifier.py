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
from sklearn import cross_validation
from sklearn.svm import SVC
from experiments.models import Hashtag


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
    if ov.isValidObject() and ov.classifiable: documents.append(ov.getVector(10))
    print i
    i+=1
    if i==200: break
    
#trainDocuments = documents[:int(len(documents)*0.80)]
#testDocuments = documents[:int(len(documents)*0.20)]
#trainX, trainy = zip(*trainDocuments)
#testX, testy = zip(*testDocuments)
#clf = SVC(probability=True)
#clf.fit(trainX, trainy)
#print clf.score(testX, testy)
