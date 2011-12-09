#'''
#Created on Dec 8, 2011
#
#@author: kykamath
#'''
#import sys, matplotlib
#sys.path.append('../')
#from library.file_io import FileIO
#from library.geo import getCenterOfMass, getHaversineDistance,\
#    plotPointsOnWorldMap, getLatticeLid, getLocationFromLid
#from settings import hashtagsWithoutEndingWindowFile,\
#    hashtagsImagesHashtagsClassFolder
#from experiments.mr_area_analysis import getOccuranesInHighestActiveRegion,\
#    LATTICE_ACCURACY, getActiveRegions, TIME_UNIT_IN_SECONDS,\
#    getOccurranceDistributionInEpochs, HashtagsClassifier,\
#    getTimeUnitsAndTimeSeries
#import numpy as np
#from itertools import combinations, groupby
#import matplotlib.pyplot as plt
#from library.stats import getOutliersRangeUsingIRQ
#from operator import itemgetter
#from library.classes import GeneralMethods
#import datetime
#
##def getStatisticsForHashtagRadius(timeRange, folderType):
##    dataX, i = [], 1
##    for hashtagObject in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(folderType,'%s_%s'%timeRange)):
##        print i;i+=1
##        occuranesInHighestActiveRegion = getOccuranesInHighestActiveRegion(hashtagObject)
##        locations = zip(*occuranesInHighestActiveRegion)[0]
##        dataX.append(getHastagRadius(locations))
##    plt.hist(dataX, bins=100)
##    plt.show()
#def plotHastagClasses(timeRange, folderType):
#    def getFileName():
#        for i in combinations('abcedfghijklmnopqrstuvwxyz',2): yield ''.join(i)+'.png'
#    count=1
#    for hashtagObject in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(folderType,'%s_%s'%timeRange)):
##        HashtagsClassifier.classify(hashtagObject)
##        if hashtagObject['h']=='ripamywinehouse':
#        print count; count+=1
#        classId = HashtagsClassifier.classify(hashtagObject)
#        if classId:
#            outputFolder = hashtagsImagesHashtagsClassFolder%folderType+'%s/%s/'%(classId, hashtagObject['h'])
#            fileNameIterator = getFileName()
#            timeUnits, timeSeries = getTimeUnitsAndTimeSeries(hashtagObject['oc'])
#            occurancesInActivityRegions = []
#            for hashtagPropagatingRegion in HashtagsClassifier._getActivityRegionsWithActivityAboveThreshold(hashtagObject):
#                validTimeUnits = [timeUnits[i] for i in range(hashtagPropagatingRegion[0], hashtagPropagatingRegion[1]+1)]
#                occurancesInActiveRegion = [(p,t) for p,t in hashtagObject['oc'] if GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) in validTimeUnits]
#                occurancesInActivityRegions.append([occurancesInActiveRegion, GeneralMethods.getRandomColor()])
#            
#            currentMainRangeId = 0
#            for occurances1, color1 in occurancesInActivityRegions:
#                outputFile=outputFolder+fileNameIterator.next();FileIO.createDirectoryForFile(outputFile)
#                print outputFile
#                ax = plt.subplot(312)
#                subRangeId = 0
#                for occurances, color in occurancesInActivityRegions:
#                    if subRangeId==currentMainRangeId: color='m'
#                    timeUnits, timeSeries = getTimeUnitsAndTimeSeries(occurances)
#                    plt.plot_date([datetime.datetime.fromtimestamp(t) for t in timeUnits], timeSeries, '-o', c=color)
#                    subRangeId+=1
#                plt.setp(ax.get_xticklabels(), rotation=10, fontsize=7)
#            
#                ax=plt.subplot(313)
#                subRangeId = 0
#                timeUnits, timeSeries = getTimeUnitsAndTimeSeries(hashtagObject['oc'])
#                plt.plot_date([datetime.datetime.fromtimestamp(t) for t in timeUnits], timeSeries, '-')
#                for occurances, color in occurancesInActivityRegions:
#                    if subRangeId==currentMainRangeId: color='m'
#                    timeUnits, timeSeries = getTimeUnitsAndTimeSeries(occurances)
#                    plt.plot_date([datetime.datetime.fromtimestamp(t) for t in timeUnits], timeSeries, '-o', c=color)
#                    subRangeId+=1
#                plt.setp(ax.get_xticklabels(), rotation=10, fontsize=7)
#                
#                plt.subplot(311)
#                occurancesGroupedByLattice = sorted(
#                                                    [(getLocationFromLid(lid.replace('_', ' ')), len(list(occs))) for lid, occs in groupby(sorted([(getLatticeLid(l, LATTICE_ACCURACY), t) for l, t in occurances1], key=itemgetter(0)), key=itemgetter(0))],
#                                                    key=itemgetter(1)
#                                                    )
#                points, colors = zip(*occurancesGroupedByLattice)
#                cm = matplotlib.cm.get_cmap('cool')
#                if len(points)>1: 
#                    sc = plotPointsOnWorldMap(points, c=colors, cmap=cm, lw=0, alpha=1.0)
#                    plt.colorbar(sc)
#                else: sc = plotPointsOnWorldMap(points, c='m', lw=0)
#                plt.title(hashtagObject['h'])
#    #            plt.show()
#                plt.savefig(outputFile); plt.clf()
#                currentMainRangeId+=1
#    #                exit()
#            
#        
#if __name__ == '__main__':
#    timeRange = (2,11)
#    folderType = 'world'
##    getStatisticsForHashtagRadius(timeRange, folderType)
#    plotHastagClasses(timeRange, folderType)