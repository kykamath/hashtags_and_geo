'''
Created on Nov 24, 2011

@author: kykamath
'''
import sys, os
import datetime
import numpy as np
from library.plotting import smooth, CurveFit
from library.kml import KML
sys.path.append('../')
from experiments.mr_area_analysis import getOccuranesInHighestActiveRegion,\
    getSourceLattice, MIN_OCCURRENCES_TO_DETERMINE_SOURCE_LATTICE,\
    HashtagsClassifier, getTimeUnitsAndTimeSeries, LATTICE_ACCURACY
from library.file_io import FileIO
import matplotlib
from settings import hashtagsAnalayzeLocalityIndexAtKFile,\
    hashtagsWithoutEndingWindowFile, hashtagSharingProbabilityGraphFile,\
    hashtagsImagesFlowInTimeForFirstNLocationsFolder,\
    hashtagsImagesFlowInTimeForFirstNOccurrencesFolder,\
    hashtagsImagesFlowInTimeForWindowOfNOccurrencesFolder,\
    hashtagsImagesTimeSeriesAnalysisFolder,\
    hashtagsWithoutEndingWindowAndOcccurencesFilteredByDistributionInTimeUnitsFile,\
    hashtagLocationTemporalClosenessGraphFile,\
    hashtagsImagesLocationClosenessFolder,\
    hashtagLocationInAndOutTemporalClosenessGraphFile,\
    hashtagsImagesLocationInfluencersFolder, hashtagsImagesGraphAnalysisFolder,\
    hashtagSharingProbabilityGraphWithTemporalClosenessFile,\
    hashtagsImagesFirstActiveTimeSeriesAnalysisFolder,\
    hashtagsImagesHashtagsClassFolder, hashtagsTrainingDataFile, hashtagsFile
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import fft, array
from collections import defaultdict
from library.graphs import plot, clusterUsingMCLClustering
import matplotlib.pyplot as plt
from operator import itemgetter
from library.geo import getHaversineDistance, plotPointsOnUSMap, getLatticeLid,\
    getLocationFromLid, plotPointsOnWorldMap
from itertools import groupby, combinations
from experiments.analysis import HashtagClass, HashtagObject
from experiments.mr_analysis import ACCURACY, getOccurranceDistributionInEpochs,\
    TIME_UNIT_IN_SECONDS, temporalScore
from library.classes import GeneralMethods
import networkx as nx

def plotHashtagsInOutGraphs(timeRange, outputFolder):
    def plotPoints(links, xlabel):
        cm = matplotlib.cm.get_cmap('cool')
        points, colors = zip(*sorted([(getLocationFromLid(k.replace('_', ' ')), v)for k, v in links.iteritems()], key=itemgetter(1)))
        sc = plotPointsOnWorldMap(points, c=colors, cmap=cm, lw=0, vmin=0, vmax=1)
        plotPointsOnWorldMap([getLocationFromLid(locationObject['id'].replace('_', ' '))], c='k', s=20, lw=0)
        plt.xlabel(xlabel), plt.colorbar(sc)
    counter=1
    for locationObject in FileIO.iterateJsonFromFile(hashtagLocationInAndOutTemporalClosenessGraphFile%(outputFolder, '%s_%s'%timeRange)): 
        point = getLocationFromLid(locationObject['id'].replace('_', ' '))
        outputFile = hashtagsImagesLocationInfluencersFolder+'%s.png'%getLatticeLid([point[1], point[0]], ACCURACY); FileIO.createDirectoryForFile(outputFile)
        print counter;counter+=1
        if not os.path.exists(outputFile):
            if locationObject['in_link'] and locationObject['out_link']:
                print outputFile
                plt.subplot(211)
                plt.title(locationObject['id'])
                plotPoints(locationObject['in_link'], 'Gets hashtags from these locations')
                plt.subplot(212)
                plotPoints(locationObject['out_link'], 'Sends hashtags to these locations')
#                plt.show()
                plt.savefig(outputFile); plt.clf()
        

def plotTimeSeriesWithHighestActiveRegion(timeRange, outputFolder):
    def plotTimeSeries(hashtagObject):
        def getDataToPlot(occ):
            occurranceDistributionInEpochs = getOccurranceDistributionInEpochs(occ)
            startEpoch, endEpoch = min(occurranceDistributionInEpochs, key=itemgetter(0))[0], max(occurranceDistributionInEpochs, key=itemgetter(0))[0]
            dataX = range(startEpoch, endEpoch, TIME_UNIT_IN_SECONDS)
            occurranceDistributionInEpochs = dict(occurranceDistributionInEpochs)
            for x in dataX: 
                if x not in occurranceDistributionInEpochs: occurranceDistributionInEpochs[x]=0
            return zip(*sorted(occurranceDistributionInEpochs.iteritems(), key=itemgetter(0)))
        
        outputFile = hashtagsImagesFirstActiveTimeSeriesAnalysisFolder%outputFolder+'%s.png'%(hashtagObject['h']); FileIO.createDirectoryForFile(outputFile)
        print unicode(outputFile).encode('utf-8')
        
        timeUnits, timeSeries = getDataToPlot(hashtagObject['oc'])
        occurencesInActiveRegion, isFirstActiveRegion = getOccuranesInHighestActiveRegion(hashtagObject, True)
        timeUnitsForActiveRegion, timeSeriesForActiveRegion = getDataToPlot(occurencesInActiveRegion)
        lid, count = getSourceLattice(hashtagObject['oc'])
        if isFirstActiveRegion and count>=MIN_OCCURRENCES_TO_DETERMINE_SOURCE_LATTICE: 
            ax=plt.subplot(211)
            plt.plot_date(map(datetime.datetime.fromtimestamp, timeUnits), timeSeries, '-')
            if not isFirstActiveRegion: plt.plot_date(map(datetime.datetime.fromtimestamp, timeUnitsForActiveRegion), timeSeriesForActiveRegion, 'o', c='r')
            else: plt.plot_date(map(datetime.datetime.fromtimestamp, timeUnitsForActiveRegion), timeSeriesForActiveRegion, 'o', c='k')
            plt.setp(ax.get_xticklabels(), rotation=30, fontsize=10)
            plt.title(hashtagObject['h'] + '(%s)'%count)
            ax=plt.subplot(212)
            plt.plot_date(map(datetime.datetime.fromtimestamp, timeUnitsForActiveRegion), timeSeriesForActiveRegion, '-')
            plt.setp(ax.get_xticklabels(), rotation=30, fontsize=10)
    #        if isFirstActiveRegion:
    #            lid, count = getSourceLattice(hashtagObject['oc'])
    #            if count>=MIN_OCCURRENCES_TO_DETERMINE_SOURCE_LATTICE:
    #                print lid, count
#            plt.show()
            plt.savefig(outputFile); 
            plt.clf()
    counter=1
    for object in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange)):
#        if object['h']=='beatohio':
        print counter; counter+=1
        plotTimeSeries(object)

def plotHastagClasses(timeRange, folderType):
    def getFileName():
        for i in combinations('abcedfghijklmnopqrstuvwxyz',2): yield ''.join(i)+'.png'
    count=1
#    for hashtagObject in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(folderType,'%s_%s'%timeRange)):
    for hashtagObject in FileIO.iterateJsonFromFile(hashtagsFile%('training_world','%s_%s'%(2,11))):
#        HashtagsClassifier.classify(hashtagObject)
        print count; count+=1
#        if hashtagObject['h']=='ripamy':
        classId = HashtagsClassifier.classify(hashtagObject)
        if classId:
            outputFile = hashtagsImagesHashtagsClassFolder%folderType+'%s/%s.png'%(classId, hashtagObject['h']); FileIO.createDirectoryForFile(outputFile)
            fileNameIterator = getFileName()
            timeUnits, timeSeries = getTimeUnitsAndTimeSeries(hashtagObject['oc'])
            occurancesInActivityRegions = [[getOccuranesInHighestActiveRegion(hashtagObject), 'm']]
#            for hashtagPropagatingRegion in HashtagsClassifier._getActivityRegionsWithActivityAboveThreshold(hashtagObject):
#                validTimeUnits = [timeUnits[i] for i in range(hashtagPropagatingRegion[0], hashtagPropagatingRegion[1]+1)]
#                occurancesInActiveRegion = [(p,t) for p,t in hashtagObject['oc'] if GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) in validTimeUnits]
#                occurancesInActivityRegions.append([occurancesInActiveRegion, GeneralMethods.getRandomColor()])
            
            currentMainRangeId = 0
            for occurances1, color1 in occurancesInActivityRegions:
#                outputFile=outputFolder+fileNameIterator.next();FileIO.createDirectoryForFile(outputFile)
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
#                plt.show()
                plt.savefig(outputFile); plt.clf()
                currentMainRangeId+=1
#                exit()

def tempAnalysis(timeRange, outputFolder):
    i = 1
    for hashtagObject in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(outputFolder,'%s_%s'%timeRange)):
        print i, HashtagsClassifier.classify(hashtagObject); i+=1
    
    
if __name__ == '__main__':
    timeRange = (2,11)
    outputFolder = 'world'
#    outputFolder='ny'
#    outputFolder='/'
#    plotHashtagFlowOnUSMap([41.046217,-73.652344], outputFolder)

#    tempAnalysis(timeRange, outputFolder)
#    plotTimeSeriesWithHighestActiveRegion(timeRange, outputFolder)
    plotHastagClasses(timeRange, outputFolder)
#    plotGraphs(timeRange, outputFolder)
#    plotNodeObject(timeRange, outputFolder)
#    plotHashtagsInOutGraphs(timeRange, outputFolder)

#    GraphAnalysis.plotConnectedComponents(timeRange, outputFolder)
#    GraphAnalysis.me(timeRange, outputFolder)
#    tempAnalysis(timeRange, outputFolder)
    
#    AnalyzeLocalityIndexAtK.LIForOccupy(timeRange)
#    AnalyzeLocalityIndexAtK.rankHashtagsBYLIScore(timeRange)
#    AnalyzeLocalityIndexAtK.plotDifferenceBetweenSourceAndLocalityLattice(timeRange)