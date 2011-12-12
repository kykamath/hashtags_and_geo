'''
Created on Nov 19, 2011

@author: kykamath
'''
import sys, datetime, matplotlib
sys.path.append('../')
from scipy import stats
from library.graphs import plot
from library.stats import getOutliersRangeUsingIRQ
from experiments.models import latticeNodeBySharingProbability,\
    latticeNodeByTemporalClosenessScore, latticeNodeByTemporalDistanceInHours,\
    latticeNodeByHaversineDistance, LatticeGraph,\
    filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance
from library.classes import GeneralMethods
from experiments.mr_area_analysis import MRAreaAnalysis, latticeIdInValidAreas,\
    LATTICE_ACCURACY, TIME_UNIT_IN_SECONDS, getSourceLattice,\
    getOccuranesInHighestActiveRegion, HashtagsClassifier
from library.geo import getHaversineDistance, getLatticeLid, getLattice,\
    getCenterOfMass, getLocationFromLid, plotPointsOnUSMap, plotPointsOnWorldMap,\
    getHaversineDistanceForLids, getLidFromLocation
from operator import itemgetter
from experiments.mr_wc import MRWC
from library.file_io import FileIO
from experiments.mr_analysis import MRAnalysis, addHashtagDisplacementsInTime,\
    getMeanDistanceBetweenLids, getMeanDistanceFromSource, getLocalityIndexAtK,\
    addSourceLatticeToHashTagObject, addHashtagLocalityIndexInTime,\
    HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS, ACCURACY,\
    HASHTAG_STARTING_WINDOW
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from library.mrjobwrapper import runMRJob
from settings import hashtagsDistributionInTimeFile, hashtagsDistributionInLatticeFile,\
    hashtagsFile, \
    hashtagsWithoutEndingWindowFile, \
    tempInputFile, inputFolder, \
    hashtagsDisplacementStatsFile, \
    hashtagsAnalayzeLocalityIndexAtKFile, hashtagWithGuranteedSourceFile,\
    hashtagsBoundarySpecificStatsFile, hashtagSharingProbabilityGraphFile,\
    hashtagsWithoutEndingWindowAndOcccurencesFilteredByDistributionInTimeUnitsFile,\
    hashtagLocationTemporalClosenessGraphFile,\
    hashtagLocationInAndOutTemporalClosenessGraphFile,\
    hashtagSharingProbabilityGraphWithTemporalClosenessFile,\
    hashtagsLatticeGraphFile, hashtagsImagesGraphAnalysisFolder,\
    hashtagsWithKnownSourceFile, hashtagsTrainingDataFile, hashtagsTestDataFile
import matplotlib.pyplot as plt
from itertools import combinations, groupby 
import numpy as np
import networkx as nx
from collections import defaultdict

def getModifiedLatticeLid(point, accuracy=0.0075):
    return '%0.1f_%0.1f'%(int(point[0]/accuracy)*accuracy, int(point[1]/accuracy)*accuracy)

def plotHashtagDistributionInTime():
    for h in FileIO.iterateJsonFromFile(hashtagsDistributionInTimeFile):
        if h['t']>300:
            distribution = dict(h['d'])
#            print unicode(h['h']).encode('utf-8'), h['t']
            ax = plt.subplot(111)
            dataX = sorted(distribution)
            plt.plot_date([datetime.datetime.fromtimestamp(e) for e in dataX], [distribution[x] for x in dataX])
            plt.title('%s (%s)'%(h['h'], h['t'])    )
            plt.setp(ax.get_xticklabels(), rotation=30, fontsize=10)
            plt.xlim(xmin=datetime.datetime(2011, 2, 1), xmax=datetime.datetime(2011, 11, 30))
            plt.show()
            
class HashtagObject:
    def __init__(self, hashtagDict): self.dict = hashtagDict
    def getLocalLattice(self): return self.dict['liAtVaryingK'][0][1]
    @staticmethod
    def iterateHashtagObjects(timeRange, file=hashtagsAnalayzeLocalityIndexAtKFile, hastagsList = None):
        for h in FileIO.iterateJsonFromFile(file%(outputFolder, '%s_%s'%timeRange)): 
            if hastagsList==None or h['h'] in hastagsList: yield HashtagObject(h)

class HashtagClass:
    hashtagsCSV = '../data/hashtags.csv'
    @staticmethod
    def createCSV(timeRange):
        i=1
        for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange)):
            FileIO.writeToFile(h['h']+', ', HashtagClass.hashtagsCSV)
    @staticmethod
    def iterateHashtagsWithClass():
        for line in FileIO.iterateLinesFromFile(HashtagClass.hashtagsCSV): 
            h, cls = line.strip().split(', ')   
            yield h.strip(), cls.strip()
    @staticmethod
    def getHashtagClasses():
        hashtags = sorted(HashtagClass.iterateHashtagsWithClass(), key=itemgetter(1))
        return dict((cls, list(h[0] for h in hashtags)) for cls, hashtags in groupby(hashtags, key=itemgetter(1)))
    
class Parameters:
    @staticmethod
    def estimateUpperRangeForTimePeriod(timeRange, outputFolder):
        dataX = []
        i=1
        for hashtagObject in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(outputFolder,'%s_%s'%timeRange)):
            print i;i+=1
            occuranesInHighestActiveRegion = getOccuranesInHighestActiveRegion(hashtagObject)
            dataX.append((occuranesInHighestActiveRegion[-1][1]-occuranesInHighestActiveRegion[0][1])/TIME_UNIT_IN_SECONDS)
        print getOutliersRangeUsingIRQ(dataX)
        plt.hist(dataX, bins=10)
        plt.show()
    @staticmethod
    def run():
        timeRange = (2,11)
        folderType = 'world'
        Parameters.estimateUpperRangeForTimePeriod(timeRange, folderType)

class HashtagsClassifierAnalysis:
    @staticmethod
    def timePeriods(timeRange, folderType):
        distribution = defaultdict(list)
#        i = 1
#        for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(folderType,'%s_%s'%timeRange)):
##            if h['h']=='jartic':
#            classId = HashtagsClassifier.classify(h)
#            if classId:
#                print i, unicode(h['h']).encode('utf-8'), classId;i+=1
#                occs = getOccuranesInHighestActiveRegion(h)
#                distribution[classId].append((occs[-1][1]-occs[0][1])/TIME_UNIT_IN_SECONDS)
#        for k,v in distribution.iteritems():
#            FileIO.writeToFileAsJson({'id':k, 'dist': v}, '../data/hashtagsClassTimePeriods.txt')
        i = 1
        for data in FileIO.iterateJsonFromFile('../data/hashtagsClassTimePeriods.txt'):
#            print data.keys()
            plt.subplot(220+i);i+=1
            plt.hist(data['dist'], bins=100)
            boundary = getOutliersRangeUsingIRQ(data['dist'])[1]
            actualHashtags = filter(lambda t:t<=boundary, data['dist'])
            meanTimePeriod = np.mean(actualHashtags)
            print {data['id'] : {'meanTimePeriod': meanTimePeriod, 'outlierBoundary': boundary}}
            plt.title(data['id']+' %0.2f %0.2f %d'%(meanTimePeriod, boundary, len(actualHashtags)))
            plt.xlim(xmax=200)
        plt.show()
class LatticeGraphPlots:
    upperRangeForTemporalDistances = 8.24972222222
    @staticmethod
    def temporalScore(lag, width):
        lag=int(lag*TIME_UNIT_IN_SECONDS)
        width=int(width*TIME_UNIT_IN_SECONDS)
        if lag==0: return 1.0
        elif lag>=width: return 0.0
        return 1-np.log(lag)/np.log(width)
    @staticmethod
    def plotLatticeSharingProbabilityOnMap(latticeGraphType, latticeObject):
        latticeObject = latticeGraphType['method'](latticeObject)
        LatticeGraph.normalizeNode(latticeObject)
        points, colors = zip(*sorted([(getLocationFromLid(neighborId.replace('_', ' ')), val) for neighborId, val in latticeObject['links'].iteritems()], key=itemgetter(1)))
        cm = matplotlib.cm.get_cmap('YlOrRd')
        sc = plotPointsOnWorldMap(points, c=colors, cmap=cm, lw = 0, vmin=0)
        plotPointsOnWorldMap([getLocationFromLid(latticeObject['id'].replace('_', ' '))], c='#00FF00', lw = 0)
        plt.xlabel(latticeGraphType['title'])
        plt.colorbar(sc)
        return sc
    @staticmethod
    def plotLatticeTemporalDistanceInHoursOnMap(latticeGraphType, latticeObject):
        latticeObject = latticeGraphType['method'](latticeObject)
        points, colors = zip(*sorted([(getLocationFromLid(neighborId.replace('_', ' ')), val) for neighborId, val in latticeObject['links'].iteritems()], key=itemgetter(1), reverse=True))
        cm = matplotlib.cm.get_cmap('autumn')
        sc = plotPointsOnWorldMap(points, c=colors, cmap=cm, lw = 0, vmin=0)
        plotPointsOnWorldMap([getLocationFromLid(latticeObject['id'].replace('_', ' '))], c='#00FF00', lw = 0)
        plt.xlabel(latticeGraphType['title'])
        plt.colorbar(sc)
        return sc
    @staticmethod
    def plotLatticeTemporalClosenessScoresOnMap(latticeGraphType, latticeObject):
        latticeObject = latticeGraphType['method'](latticeObject)
        LatticeGraph.normalizeNode(latticeObject)
        points, colors = zip(*sorted([(getLocationFromLid(neighborId.replace('_', ' ')), val) for neighborId, val in latticeObject['links'].iteritems()], key=itemgetter(1)))
        cm = matplotlib.cm.get_cmap('YlOrRd')
        sc = plotPointsOnWorldMap(points, c=colors, cmap=cm, lw = 0, vmin=0)
        plotPointsOnWorldMap([getLocationFromLid(latticeObject['id'].replace('_', ' '))], c='#00FF00', lw = 0)
        plt.xlabel(latticeGraphType['title'])
        plt.colorbar(sc)
        return sc
    @staticmethod
    def plotSharingProbabilityAndTemporalDistanceInHoursOnMap(timeRange, outputFolder):
        i = 1
        for latticeObject in FileIO.iterateJsonFromFile(hashtagsLatticeGraphFile%(outputFolder,'%s_%s'%timeRange)):
            latticePoint = getLocationFromLid(latticeObject['id'].replace('_', ' '))
            latticeId = getLatticeLid([latticePoint[1], latticePoint[0]], LATTICE_ACCURACY)
            plt.subplot(211)
            plt.title(latticeId)
            LatticeGraphPlots.plotLatticeSharingProbabilityOnMap(LatticeGraph.typeSharingProbability, latticeObject)
            plt.subplot(212)
            LatticeGraphPlots.plotLatticeTemporalDistanceInHoursOnMap(LatticeGraph.typeTemporalDistanceInHours, latticeObject)
#            plt.show()
            outputFile = hashtagsImagesGraphAnalysisFolder%outputFolder+'%s_and_%s/%s.png'%(LatticeGraph.typeSharingProbability['id'], LatticeGraph.typeTemporalDistanceInHours['id'], latticeId); FileIO.createDirectoryForFile(outputFile)
            print i, outputFile; i+=1
            plt.savefig(outputFile); plt.clf()
    @staticmethod
    def plotSharingProbabilityAndTemporalClosenessScoresOnMap(timeRange, outputFolder):
        i = 1
        for latticeObject in FileIO.iterateJsonFromFile(hashtagsLatticeGraphFile%(outputFolder,'%s_%s'%timeRange)):
            latticePoint = getLocationFromLid(latticeObject['id'].replace('_', ' '))
            latticeId = getLatticeLid([latticePoint[1], latticePoint[0]], LATTICE_ACCURACY)
            plt.subplot(211)
            plt.title(latticeId)
            LatticeGraphPlots.plotLatticeSharingProbabilityOnMap(LatticeGraph.typeSharingProbability, latticeObject)
            plt.subplot(212)
            LatticeGraphPlots.plotLatticeTemporalClosenessScoresOnMap(LatticeGraph.typeTemporalCloseness, latticeObject)
            plt.show()
            outputFile = hashtagsImagesGraphAnalysisFolder%outputFolder+'%s_and_%s/%s.png'%(LatticeGraph.typeSharingProbability['id'], LatticeGraph.typeTemporalCloseness['id'], latticeId); FileIO.createDirectoryForFile(outputFile)
            print i, outputFile; i+=1
#            plt.savefig(outputFile); plt.clf()
    @staticmethod
    def plotLatticesOnMap(timeRange, outputFolder):
        points = [getLocationFromLid(latticeObject['id'].replace('_', ' ')) for latticeObject in FileIO.iterateJsonFromFile(hashtagsLatticeGraphFile%(outputFolder,'%s_%s'%timeRange))]
        plotPointsOnWorldMap(points, c='m', lw=0)
        plt.show()
    @staticmethod
    def determineUpperRangeForTemporalDistances(timeRange, outputFolder):
        i = 1
        temporalDistancesForAllLattices = []
        for latticeObject in FileIO.iterateJsonFromFile(hashtagsLatticeGraphFile%(mrOutputFolder,'%s_%s'%timeRange)):
            print i, latticeObject['id']; i+=1
            for neighborLattice, neighborHashtags in latticeObject['links'].iteritems():
                temporalDistancesForAllLattices+=zip(*filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags).iteritems())[1]
#            if i==10: break
        print getOutliersRangeUsingIRQ(temporalDistancesForAllLattices)[1]
    @staticmethod
    def measureCorrelations(timeRange, outputFolder):
        '''
        ['haversine_distance', 'temporal_distance_in_hours', 0.20147108648121248]
        ['haversine_distance', 'sharing_probability', -0.19587239643328627]
        '''
        measures = [
                    (LatticeGraph.typeHaversineDistance, LatticeGraph.typeTemporalDistanceInHours),
                    (LatticeGraph.typeHaversineDistance, LatticeGraph.typeSharingProbability),
                    ]
        runData = []
        for xMeasure, yMeasure in measures:
            i, xdata, ydata = 1, [], []
            for latticeObject in FileIO.iterateJsonFromFile(hashtagsLatticeGraphFile%(outputFolder,'%s_%s'%timeRange)):
                print i, latticeObject['id']; i+=1
                xdata+=zip(*xMeasure['method'](latticeObject)['links'].iteritems())[1]
                ydata+=zip(*yMeasure['method'](latticeObject)['links'].iteritems())[1]
#                if i==200: break
            preasonsCorrelation, _ = stats.pearsonr(xdata, ydata)
#            plt.scatter(xdata[:5000], ydata[:5000])
#            plt.title('Pearson\'s co-efficient %0.3f'%preasonsCorrelation)
#            plt.xlabel(xMeasure['title']), plt.ylabel(yMeasure['title'])
#            plt.show()
            runData.append([xMeasure['id'], yMeasure['id'], preasonsCorrelation])
        for i in runData:
            print i
    @staticmethod
    def run(timeRange, outputFolder):
#        LatticeGraph.plotLatticesOnMap(timeRange, mrOutputFolder)
#        LatticeGraph.determineUpperRangeForTemporalDistanceScores(timeRange, outputFolder)
#        LatticeGraph.plotSharingProbabilityAndTemporalDistanceInHoursOnMap(timeRange, outputFolder)
        LatticeGraphPlots.plotSharingProbabilityAndTemporalClosenessScoresOnMap(timeRange, outputFolder)
#        LatticeGraph.measureCorrelations(timeRange, outputFolder)

def plotHashtagSourcesOnMap(timeRange, outputFolder):
    i = 1
    distribution = defaultdict(int)
    for hashtagObject in FileIO.iterateJsonFromFile(hashtagsWithKnownSourceFile%(outputFolder,'%s_%s'%timeRange)):
        occuranesInHighestActiveRegion, isFirstActiveRegion = getOccuranesInHighestActiveRegion(hashtagObject, True)
        source, count = getSourceLattice(occuranesInHighestActiveRegion)
        print i, source;i+=1
        distribution[getLidFromLocation(source)]+=1
#        if i==10: break
    points, colors = zip(*[(getLocationFromLid(k),v) for k, v in sorted(distribution.iteritems(), key=itemgetter(1))])
    cm = matplotlib.cm.get_cmap('Paired')
    sc = plotPointsOnWorldMap(points, c=colors, cmap=cm, lw = 0)
    plt.colorbar(sc)
    plt.show()


def tempAnalysis(timeRange, mrOutputFolder):
    i = 1
#    temporalDistancesForAllLattices = []
    measures = [(LatticeGraph.typeSharingProbability, LatticeGraph.typeTemporalDistanceInHours)]
    for latticeObject in FileIO.iterateJsonFromFile(hashtagsLatticeGraphFile%(mrOutputFolder,'%s_%s'%timeRange)):
        latticeObject = latticeNodeByHashtagDiffusionLocationVisitation(latticeObject, generateTemporalClosenessScore=True)
        print sum(latticeObject['links']['out'].values())
        LatticeGraph.normalizeNode(latticeObject)
        print sum(latticeObject['links']['out'].values())
        exit()
        
#    mra = MRAnalysis()
#    for h in FileIO.iterateJsonFromFile('/mnt/chevron/kykamath/data/geo/hashtags/analysis/world/2_11/hashtagsWithoutEndingWindow'):
#        mra.buildHashtagTemporalClosenessGraphMap(None, h)
#        exit()

def getInputFiles(months, folderType='/'): return [inputFolder+folderType+'/'+str(m) for m in months]        
#def mr_analysis(timeRange, outputFolder):
##    runMRJob(MRAnalysis, hashtagsFile%(outputFolder, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), outputFolder), jobconf={'mapred.reduce.tasks':300})
##    runMRJob(MRAnalysis, hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), outputFolder), jobconf={'mapred.reduce.tasks':160})
##    runMRJob(MRAnalysis, hashtagsDistributionInTimeFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
##    runMRJob(MRAnalysis, hashtagsDistributionInLatticeFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
##    runMRJob(MRAnalysis, hashtagsCenterOfMassAnalysisWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1)), jobconf={'mapred.reduce.tasks':300})
##    runMRJob(MRAnalysis, hashtagsSpreadInTimeFile%(outputFolder, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1)), jobconf={'mapred.reduce.tasks':300})
##    runMRJob(MRAnalysis, hashtagsDisplacementStatsFile%(outputFolder, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), outputFolder), jobconf={'mapred.reduce.tasks':90})
##    runMRJob(MRAnalysis, hashtagsAnalayzeLocalityIndexAtKFile%(outputFolder, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1)), jobconf={'mapred.reduce.tasks':300})
##    runMRJob(MRAnalysis, hashtagWithGuranteedSourceFile%(outputFolder, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1)), jobconf={'mapred.reduce.tasks':300})
##    runMRJob(MRAnalysis, hashtagsBoundarySpecificStatsFile%(outputFolder,'%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), outputFolder), jobconf={'mapred.reduce.tasks':600})
##    runMRJob(MRAnalysis, hashtagSharingProbabilityGraphFile%(outputFolder,'%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), outputFolder), jobconf={'mapred.reduce.tasks':160})
##    runMRJob(MRAnalysis, hashtagLocationTemporalClosenessGraphFile%(outputFolder,'%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), outputFolder), jobconf={'mapred.reduce.tasks':160})
#    runMRJob(MRAnalysis, hashtagLocationInAndOutTemporalClosenessGraphFile%(outputFolder,'%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), outputFolder), jobconf={'mapred.reduce.tasks':160})

def mr_area_analysis(timeRange, folderType, mrOutputFolder):
    runMRJob(MRAreaAnalysis, hashtagsTrainingDataFile%(mrOutputFolder,'%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), folderType), jobconf={'mapred.reduce.tasks':160})
#    runMRJob(MRAreaAnalysis, hashtagsWithoutEndingWindowFile%(mrOutputFolder,'%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), folderType), jobconf={'mapred.reduce.tasks':160})
#    runMRJob(MRAreaAnalysis, hashtagsWithKnownSourceFile%(mrOutputFolder,'%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), folderType), jobconf={'mapred.reduce.tasks':160})
#    runMRJob(MRAreaAnalysis, hashtagsLatticeGraphFile%(mrOutputFolder,'%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), folderType), jobconf={'mapred.reduce.tasks':160})
#    runMRJob(MRAreaAnalysis, hashtagLocationTemporalClosenessGraphFile%(mrOutputFolder,'%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), folderType), jobconf={'mapred.reduce.tasks':160})


if __name__ == '__main__':
    timeRange = (2,11)
#    outputFolder = '/'
#    outputFolder = 'world'
#    mr_analysis(timeRange, outputFolder)
    
    folderType = 'world'
#    folderType = '/'
    mrOutputFolder = 'world'
    mr_area_analysis(timeRange, folderType, mrOutputFolder)
#    HashtagsClassifierAnalysis.timePeriods(timeRange, folderType)

#    tempAnalysis(timeRange, mrOutputFolder)

#    plotHashtagSourcesOnMap(timeRange, mrOutputFolder)
#    LatticeGraph.run(timeRange, mrOutputFolder)
#    Parameters.run()
    
    