'''
Created on Nov 19, 2011

@author: kykamath
'''
import sys, datetime, matplotlib
from library.graphs import plot
sys.path.append('../')
from library.classes import GeneralMethods
from experiments.mr_area_analysis import MRAreaAnalysis, latticeIdInValidAreas,\
    LATTICE_ACCURACY
from library.geo import getHaversineDistance, getLatticeLid, getLattice,\
    getCenterOfMass, getLocationFromLid, plotPointsOnUSMap, plotPointsOnWorldMap
from operator import itemgetter
from experiments.mr_wc import MRWC
from library.file_io import FileIO
from experiments.mr_analysis import MRAnalysis, addHashtagDisplacementsInTime,\
    getMeanDistanceBetweenLids, getMeanDistanceFromSource, getLocalityIndexAtK,\
    addSourceLatticeToHashTagObject, addHashtagLocalityIndexInTime,\
    HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS, ACCURACY,\
    HASHTAG_STARTING_WINDOW,\
    getOccuranesInHighestActiveRegion
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    hashtagsLatticeGraphFile, hashtagsImagesGraphAnalysisFolder
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

def latticeNodeBySharingProbability(latticeObject):
    latticeHashtagsSet = set(latticeObject['hashtags'])
    dataToReturn = {'id': latticeObject['id'], 'links': {}}
    for neighborLattice, neighborHashtags in latticeObject['links'].iteritems():
        neighborHashtagsSet = set(neighborHashtags)
        dataToReturn['links'][neighborLattice]=len(latticeHashtagsSet.intersection(neighborHashtagsSet))/float(len(latticeHashtagsSet))
    return dataToReturn
    
class LatticeGraph:
    typeSharingProbability = {'id': 'sharing_probability', 'method': latticeNodeBySharingProbability, 'title': 'Probability of sharing hastags'}
    def __init__(self, graphFile, latticeGraphType, graphType=nx.Graph):
        self.graphFile = graphFile
        self.latticeGraphType = latticeGraphType
        self.graphType = graphType
    def load(self):
#        i = 1
        self.graph = self.graphType()
        for latticeObject in FileIO.iterateJsonFromFile(self.graphFile):
            latticeObject = self.latticeGraphType['method'](latticeObject)
            for no, w in latticeObject['links'].iteritems(): self.graph.add_edge(latticeObject['id'], no, {'w': w})
#            print i; i+=1; 
#            if i==10: break
        return self.graph
    def plotSharingProbabilityGraphsOnMap(self):
        for latticeObject in FileIO.iterateJsonFromFile(self.graphFile):
            latticeObject = self.latticeGraphType['method'](latticeObject)
            points, colors = zip(*sorted([(getLocationFromLid(neighborId.replace('_', ' ')), val) for neighborId, val in latticeObject['links'].iteritems()], key=itemgetter(1)))
            ax = plt.subplot(111)
            cm = matplotlib.cm.get_cmap('YlOrRd')
            sc = plotPointsOnWorldMap(points, c=colors, cmap=cm, lw = 0, vmin=0, vmax=1)
            latticePoint = getLocationFromLid(latticeObject['id'].replace('_', ' '))
            plotPointsOnWorldMap([latticePoint], c='#00FF00', lw = 0)
            plt.xlabel(self.typeSharingProbability['title'])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(sc, cax=cax)
#            plt.show()
            outputFile = hashtagsImagesGraphAnalysisFolder%self.graphFile.split('/')[-3]+'%s/%s.png'%(getLatticeLid([latticePoint[1], latticePoint[0]], LATTICE_ACCURACY), self.latticeGraphType['id']); FileIO.createDirectoryForFile(outputFile)
            print outputFile
            plt.savefig(outputFile); plt.clf()
    @staticmethod
    def plotLatticesOnMap(timeRange, outputFolder):
        points = [getLocationFromLid(latticeObject['id'].replace('_', ' ')) for latticeObject in FileIO.iterateJsonFromFile(hashtagsLatticeGraphFile%(outputFolder,'%s_%s'%timeRange))]
        plotPointsOnWorldMap(points, c='m', lw=0)
        plt.show()
    @staticmethod
    def plotGraphsOnMap(timeRange, outputFolder):
        LatticeGraph(hashtagsLatticeGraphFile%(outputFolder,'%s_%s'%timeRange), LatticeGraph.typeSharingProbability).plotSharingProbabilityGraphsOnMap()

def tempAnalysis(timeRange, mrOutputFolder):
    LatticeGraph(hashtagsLatticeGraphFile%(mrOutputFolder,'%s_%s'%timeRange), LatticeGraph.typeSharingProbability).plotSharingProbabilityGraphsOnMap()
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
#    runMRJob(MRAreaAnalysis, hashtagsWithoutEndingWindowFile%(mrOutputFolder,'%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), folderType), jobconf={'mapred.reduce.tasks':160})
    runMRJob(MRAreaAnalysis, hashtagsLatticeGraphFile%(mrOutputFolder,'%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), folderType), jobconf={'mapred.reduce.tasks':160})
#    runMRJob(MRAreaAnalysis, hashtagLocationTemporalClosenessGraphFile%(mrOutputFolder,'%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), folderType), jobconf={'mapred.reduce.tasks':160})


if __name__ == '__main__':
    timeRange = (2,11)
#    outputFolder = '/'
#    outputFolder = 'world'
#    mr_analysis(timeRange, outputFolder)
    
    folderType = 'world'
#    folderType = '/'
    mrOutputFolder = 'world'
#    mr_area_analysis(timeRange, folderType, mrOutputFolder)

#    tempAnalysis(timeRange, mrOutputFolder)
#    LatticeGraph.plotLatticesOnMap(timeRange, mrOutputFolder)
    LatticeGraph.plotGraphsOnMap(timeRange, mrOutputFolder)
    
    