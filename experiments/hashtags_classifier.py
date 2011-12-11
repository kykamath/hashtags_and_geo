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

CLASSIFIER_TIME_UNIT_IN_SECONDS = 5*60

timeRange, folderType = (2,11), 'world'

def normalize(data):
    total = math.sqrt(float(sum([d**2 for d in data])))
    if total==0: return map(lambda d: 0, data)
    return map(lambda d: d/total, data)
class OccuranceVector:
    def __init__(self, hashtagObject):
        self.hashtagObject = hashtagObject
        occurranceDistributionInEpochs = getOccurranceDistributionInEpochs(getOccuranesInHighestActiveRegion(self.hashtagObject), timeUnit=CLASSIFIER_TIME_UNIT_IN_SECONDS, fillInGaps=True, occurancesCount=False)
        self.occurances = zip(*sorted(occurranceDistributionInEpochs.iteritems(), key=itemgetter(0)))[1]
        self.occuranceCountVector = map(lambda t: len(t), self.occurances)
        self.occuranceLatticesVector = []
        for t in self.occurances:
            if t: self.occuranceLatticesVector.append(getRadius(zip(*t)[0]))
            else: self.occuranceLatticesVector.append(0)
    def getVector(self, length): return normalize(self.occuranceCountVector[:length]) + self.occuranceLatticesVector[:length]
        
i = 1
for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(folderType,'%s_%s'%timeRange)):
#    occurranceDistributionInEpochs = getOccurranceDistributionInEpochs(getOccuranesInHighestActiveRegion(h), timeUnit=CLASSIFIER_TIME_UNIT_IN_SECONDS, fillInGaps=True)
#    occuranceVector = zip(*sorted(occurranceDistributionInEpochs.iteritems(), key=itemgetter(0)))[1]
    ov = OccuranceVector(h)
    print i, len(h['oc']), HashtagsClassifier.classify(h), ov.getVector(5)
    i+=1
    if i==5: exit()