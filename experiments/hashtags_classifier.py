'''
Created on Dec 8, 2011

@author: kykamath
'''
import sys, matplotlib
sys.path.append('../')
from library.file_io import FileIO
from library.geo import getCenterOfMass, getHaversineDistance,\
    plotPointsOnWorldMap, getLatticeLid, getLocationFromLid
from settings import hashtagsWithoutEndingWindowFile,\
    hashtagsImagesHashtagsClassFolder
from experiments.mr_area_analysis import getOccuranesInHighestActiveRegion,\
    LATTICE_ACCURACY, getActiveRegions, TIME_UNIT_IN_SECONDS
import numpy as np
from itertools import combinations, groupby
import matplotlib.pyplot as plt
from library.stats import getOutliersRangeUsingIRQ
from experiments.mr_analysis import getOccurranceDistributionInEpochs
from operator import itemgetter
from library.classes import GeneralMethods
import datetime

def getTimeUnitsAndTimeSeries(occurences):
    occurranceDistributionInEpochs = getOccurranceDistributionInEpochs(occurences)
    startEpoch, endEpoch = min(occurranceDistributionInEpochs, key=itemgetter(0))[0], max(occurranceDistributionInEpochs, key=itemgetter(0))[0]
    dataX = range(startEpoch, endEpoch, TIME_UNIT_IN_SECONDS)
    occurranceDistributionInEpochs = dict(occurranceDistributionInEpochs)
    for x in dataX: 
        if x not in occurranceDistributionInEpochs: occurranceDistributionInEpochs[x]=0
    return zip(*sorted(occurranceDistributionInEpochs.iteritems(), key=itemgetter(0)))
class HashtagsClassifier:
    PERIODICITY_ID_SLOW_BURST = 'slow_burst'
    PERIODICITY_ID_SUDDEN_BURST = 'sudden_burst'
    PERIODICITY_ID_PERIODIC = 'periodic' 

    LOCALITY_ID_LOCAL = 'local'
    LOCALITY_ID_LOCAL_SAME_PLACE = 'local_same_place'
    LOCALITY_ID_LOCAL_DIFF_PLACE = 'local_diff_place'
    LOCALITY_ID_NON_LOCAL = 'non_local'
    
    RADIUS_LIMIT_FOR_LOCAL_HASHTAG_IN_MILES=500
    PERCENTAGE_OF_OCCURANCES_IN_SUB_ACTIVITY_REGION=0.60
    @staticmethod
    def getId(locality, periodicity): return '%s_::_%s'%(periodicity, locality)
    @staticmethod
    def classify(hashtagObject): 
        periodicityId = HashtagsClassifier.getPeriodicityClass(hashtagObject)
        if periodicityId!=HashtagsClassifier.PERIODICITY_ID_PERIODIC: return HashtagsClassifier.getId(HashtagsClassifier.getHastagLocalityClassForHighestActivityPeriod(hashtagObject), periodicityId)
        else: return HashtagsClassifier.getId(HashtagsClassifier.getHastagLocalityClassForAllActivityPeriod(hashtagObject), periodicityId)
    @staticmethod
    def getHastagLocalityClassForHighestActivityPeriod(hashtagObject): 
        occuranesInHighestActiveRegion = getOccuranesInHighestActiveRegion(hashtagObject)
        locations = zip(*occuranesInHighestActiveRegion)[0]
        meanLid = getCenterOfMass(locations,accuracy=LATTICE_ACCURACY)
        distances = [getHaversineDistance(meanLid, p) for p in locations]
        _, upperBoundForDistance = getOutliersRangeUsingIRQ(distances)
        if np.mean(filter(lambda d: d<=upperBoundForDistance, distances)) >= HashtagsClassifier.RADIUS_LIMIT_FOR_LOCAL_HASHTAG_IN_MILES: return HashtagsClassifier.LOCALITY_ID_NON_LOCAL
        else: return HashtagsClassifier.LOCALITY_ID_LOCAL
    @staticmethod
    def getHastagLocalityClassForAllActivityPeriod(hashtagObject):
        timeUnits, timeSeries = getTimeUnitsAndTimeSeries(hashtagObject['oc'])
        occurancesInActivityRegions = []
        for hashtagPropagatingRegion in HashtagsClassifier._getActivityRegionsWithActivityAboveThreshold(hashtagObject):
            validTimeUnits = [timeUnits[i] for i in range(hashtagPropagatingRegion[0], hashtagPropagatingRegion[1]+1)]
            occurancesInActiveRegion = [(p,t) for p,t in hashtagObject['oc'] if GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) in validTimeUnits]
            occurancesInActivityRegions.append(occurancesInActiveRegion)
        activityPeriodSpecificMean = []
        for currentOccurences in occurancesInActivityRegions:
            locations = zip(*currentOccurences)[0]
            meanLid = getCenterOfMass(locations,accuracy=LATTICE_ACCURACY)
            distances = [getHaversineDistance(meanLid, p) for p in locations]
            _, upperBoundForDistance = getOutliersRangeUsingIRQ(distances)
            if np.mean(filter(lambda d: d<=upperBoundForDistance, distances)) >= HashtagsClassifier.RADIUS_LIMIT_FOR_LOCAL_HASHTAG_IN_MILES: return HashtagsClassifier.LOCALITY_ID_NON_LOCAL
            else: activityPeriodSpecificMean.append(meanLid)
        meanLid = getCenterOfMass(activityPeriodSpecificMean,accuracy=LATTICE_ACCURACY)
        distances = [getHaversineDistance(meanLid, p) for p in activityPeriodSpecificMean]
        if np.mean(filter(lambda d: d<=upperBoundForDistance, distances)) >= HashtagsClassifier.RADIUS_LIMIT_FOR_LOCAL_HASHTAG_IN_MILES: return HashtagsClassifier.LOCALITY_ID_LOCAL_DIFF_PLACE
        else: return HashtagsClassifier.LOCALITY_ID_LOCAL_SAME_PLACE
        
    @staticmethod
    def _getActivityRegionsWithActivityAboveThreshold(hashtagObject):
        occurranceDistributionInEpochs = getOccurranceDistributionInEpochs(hashtagObject['oc'])
        startEpoch, endEpoch = min(occurranceDistributionInEpochs, key=itemgetter(0))[0], max(occurranceDistributionInEpochs, key=itemgetter(0))[0]
        dataX = range(startEpoch, endEpoch, TIME_UNIT_IN_SECONDS)
        occurranceDistributionInEpochs = dict(occurranceDistributionInEpochs)
        for x in dataX: 
            if x not in occurranceDistributionInEpochs: occurranceDistributionInEpochs[x]=0
        timeUnits, timeSeries = zip(*sorted(occurranceDistributionInEpochs.iteritems(), key=itemgetter(0)))
        _, _, sizeOfMaxActivityRegion = max(getActiveRegions(timeSeries), key=itemgetter(2))
        activityRegionsWithActivityAboveThreshold=[]
        for start, end, size in getActiveRegions(timeSeries):
            if size>=HashtagsClassifier.PERCENTAGE_OF_OCCURANCES_IN_SUB_ACTIVITY_REGION*sizeOfMaxActivityRegion: activityRegionsWithActivityAboveThreshold.append([start, end, size])
        return activityRegionsWithActivityAboveThreshold
    @staticmethod
    def getPeriodicityClass(hashtagObject):
        occurranceDistributionInEpochs = getOccurranceDistributionInEpochs(hashtagObject['oc'])
        startEpoch, endEpoch = min(occurranceDistributionInEpochs, key=itemgetter(0))[0], max(occurranceDistributionInEpochs, key=itemgetter(0))[0]
        dataX = range(startEpoch, endEpoch, TIME_UNIT_IN_SECONDS)
        occurranceDistributionInEpochs = dict(occurranceDistributionInEpochs)
        for x in dataX: 
            if x not in occurranceDistributionInEpochs: occurranceDistributionInEpochs[x]=0
        timeUnits, timeSeries = zip(*sorted(occurranceDistributionInEpochs.iteritems(), key=itemgetter(0)))
        _, _, sizeOfMaxActivityRegion = max(getActiveRegions(timeSeries), key=itemgetter(2))
        activityRegionsWithActivityAboveThreshold=[]
        for start, end, size in getActiveRegions(timeSeries):
            if size>=HashtagsClassifier.PERCENTAGE_OF_OCCURANCES_IN_SUB_ACTIVITY_REGION*sizeOfMaxActivityRegion: activityRegionsWithActivityAboveThreshold.append([start, end, size]) 
        if len(activityRegionsWithActivityAboveThreshold)>1: return HashtagsClassifier.PERIODICITY_ID_PERIODIC
        else:
            hashtagPropagatingRegion = activityRegionsWithActivityAboveThreshold[0]
            validTimeUnits = [timeUnits[i] for i in range(hashtagPropagatingRegion[0], hashtagPropagatingRegion[1]+1)]
            if GeneralMethods.approximateEpoch(hashtagObject['oc'][0][1], TIME_UNIT_IN_SECONDS)==validTimeUnits[0]: return HashtagsClassifier.PERIODICITY_ID_SUDDEN_BURST
            return HashtagsClassifier.PERIODICITY_ID_SLOW_BURST
        
#def getStatisticsForHashtagRadius(timeRange, folderType):
#    dataX, i = [], 1
#    for hashtagObject in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(folderType,'%s_%s'%timeRange)):
#        print i;i+=1
#        occuranesInHighestActiveRegion = getOccuranesInHighestActiveRegion(hashtagObject)
#        locations = zip(*occuranesInHighestActiveRegion)[0]
#        dataX.append(getHastagRadius(locations))
#    plt.hist(dataX, bins=100)
#    plt.show()
def plotHastagClasses(timeRange, folderType):
    def getFileName():
        for i in combinations('abcdefghijklmnopqrstuvwxyz',2): yield ''.join(i)+'.png'
    count=1
    for hashtagObject in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(folderType,'%s_%s'%timeRange)):
#        HashtagsClassifier.classify(hashtagObject)
        if hashtagObject['h']=='happybdayflea':
            print count; count+=1
            classId = HashtagsClassifier.classify(hashtagObject)
            outputFolder = hashtagsImagesHashtagsClassFolder%folderType+'%s/%s/'%(classId, hashtagObject['h'])
            fileNameIterator = getFileName()
            timeUnits, timeSeries = getTimeUnitsAndTimeSeries(hashtagObject['oc'])
            occurancesInActivityRegions = []
            for hashtagPropagatingRegion in HashtagsClassifier._getActivityRegionsWithActivityAboveThreshold(hashtagObject):
                validTimeUnits = [timeUnits[i] for i in range(hashtagPropagatingRegion[0], hashtagPropagatingRegion[1]+1)]
                occurancesInActiveRegion = [(p,t) for p,t in hashtagObject['oc'] if GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) in validTimeUnits]
                occurancesInActivityRegions.append([occurancesInActiveRegion, GeneralMethods.getRandomColor()])
            
            currentMainRangeId = 0
            for occurances1, color1 in occurancesInActivityRegions:
                outputFile=outputFolder+fileNameIterator.next();FileIO.createDirectoryForFile(outputFile)
                print outputFile
                ax = plt.subplot(312)
                subRangeId = 0
                for occurances, color in occurancesInActivityRegions:
                    if subRangeId==currentMainRangeId: color='m'
                    timeUnits, timeSeries = getTimeUnitsAndTimeSeries(occurances)
                    plt.plot_date([datetime.datetime.fromtimestamp(t) for t in timeUnits], timeSeries, '-o', c=color)
                    subRangeId+=1
                plt.setp(ax.get_xticklabels(), rotation=10, fontsize=7)
            
                ax=plt.subplot(313)
                subRangeId = 0
                timeUnits, timeSeries = getTimeUnitsAndTimeSeries(hashtagObject['oc'])
                plt.plot_date([datetime.datetime.fromtimestamp(t) for t in timeUnits], timeSeries, '-')
                for occurances, color in occurancesInActivityRegions:
                    if subRangeId==currentMainRangeId: color='m'
                    timeUnits, timeSeries = getTimeUnitsAndTimeSeries(occurances)
                    plt.plot_date([datetime.datetime.fromtimestamp(t) for t in timeUnits], timeSeries, '-o', c=color)
                    subRangeId+=1
                plt.setp(ax.get_xticklabels(), rotation=10, fontsize=7)
                
                plt.subplot(311)
                occurancesGroupedByLattice = sorted(
                                                    [(getLocationFromLid(lid.replace('_', ' ')), len(list(occs))) for lid, occs in groupby(sorted([(getLatticeLid(l, LATTICE_ACCURACY), t) for l, t in occurances1], key=itemgetter(0)), key=itemgetter(0))],
                                                    key=itemgetter(1)
                                                    )
                points, colors = zip(*occurancesGroupedByLattice)
                cm = matplotlib.cm.get_cmap('cool')
                if len(points)>1: 
                    sc = plotPointsOnWorldMap(points, c=colors, cmap=cm, lw=0, alpha=1.0)
                    plt.colorbar(sc)
                else: sc = plotPointsOnWorldMap(points, c='m', lw=0)
                plt.title(hashtagObject['h'])
    #            plt.show()
                plt.savefig(outputFile); plt.clf()
                currentMainRangeId+=1
                exit()
            
        
if __name__ == '__main__':
    timeRange = (2,11)
    folderType = 'world'
#    getStatisticsForHashtagRadius(timeRange, folderType)
    plotHastagClasses(timeRange, folderType)