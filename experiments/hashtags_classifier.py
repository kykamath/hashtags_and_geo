'''
Created on Dec 8, 2011

@author: kykamath
'''
from library.file_io import FileIO
from library.geo import getCenterOfMass, getHaversineDistance
from settings import hashtagsWithoutEndingWindowFile
from experiments.mr_area_analysis import getOccuranesInHighestActiveRegion,\
    LATTICE_ACCURACY
import numpy as np
import matplotlib.pyplot as plt
from library.stats import getOutliersRangeUsingIRQ

def getHastagRadius(llids): 
    meanLid = getCenterOfMass(llids,accuracy=LATTICE_ACCURACY)
    distances = [getHaversineDistance(meanLid, p) for p in llids]
    _, upperBoundForDistance = getOutliersRangeUsingIRQ(distances)
    return np.mean(filter(lambda d: d<=upperBoundForDistance, distances))
class HashtagsClassifier:
    RADIUS_LIMIT_FOR_LOCAL_HASHTAG_IN_MILES=500
    @staticmethod
    def classify(hashtagObject):
        occuranesInHighestActiveRegion = getOccuranesInHighestActiveRegion(hashtagObject)
        locations = zip(*occuranesInHighestActiveRegion)[0]
        print hashtagObject['h']
        print getHastagRadius(locations)
        exit()
        
def getStatisticsForHashtagRadius(timeRange, folderType):
    dataX, i = [], 1
    for hashtagObject in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(folderType,'%s_%s'%timeRange)):
        print i;i+=1
        occuranesInHighestActiveRegion = getOccuranesInHighestActiveRegion(hashtagObject)
        locations = zip(*occuranesInHighestActiveRegion)[0]
        dataX.append(getHastagRadius(locations))
    plt.hist(dataX, bins=100)
    plt.show()
        
        
if __name__ == '__main__':
    timeRange = (2,11)
    folderType = 'world'
    getStatisticsForHashtagRadius(timeRange, folderType)
#    for hashtagObject in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(folderType,'%s_%s'%timeRange)):
#        HashtagsClassifier.classify(hashtagObject)