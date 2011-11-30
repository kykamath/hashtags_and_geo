'''
Created on Nov 24, 2011

@author: kykamath
'''
import sys, os
sys.path.append('../')
from library.file_io import FileIO
import matplotlib
from settings import hashtagsAnalayzeLocalityIndexAtKFile,\
    hashtagsWithoutEndingWindowFile, hashtagSharingProbabilityGraphFile,\
    hashtagsImagesHastagsSharingProbabilitiesFolder,\
    hashtagsImagesFlowInTimeForFirstNLocationsFolder,\
    hashtagsImagesFlowInTimeForFirstNOccurrencesFolder,\
    hashtagsImagesFlowInTimeForWindowOfNOccurrencesFolder
from library.graphs import plot
import matplotlib.pyplot as plt
from operator import itemgetter
from library.geo import getHaversineDistance, plotPointsOnUSMap, getLatticeLid,\
    getLocationFromLid, plotPointsOnWorldMap
from itertools import groupby
from experiments.analysis import HashtagClass, HashtagObject
from experiments.mr_analysis import ACCURACY
from library.classes import GeneralMethods
import networkx as nx

class AnalyzeLocalityIndexAtK:
    @staticmethod
    def LIForOccupy(timeRange):
        imagesFolder = '/tmp/images/'
        occupyList = ['occupywallst', 'occupyoakland', 'occupydc', 'occupysf']
        FileIO.createDirectoryForFile(imagesFolder+'dsf')
        for h in FileIO.iterateJsonFromFile(hashtagsAnalayzeLocalityIndexAtKFile%'%s_%s'%timeRange):
    #        if h['h'].startswith('occupy') or h['h']=='ows':
            if h['h'] in occupyList:
                print h['h']
                dataX, dataY = zip(*[(k, v[0])for k, v in h['liAtVaryingK']])
                plt.plot(dataX, dataY, label=h['h'])
    #    plt.title(h['h'])
        plt.ylim(ymax=4000)
        plt.legend(loc=2)
        plt.show()
    @staticmethod
    def rankHashtagsBYLIScore(timeRange):
        hashtagsLIScore = [(h['h'], h['liAtVaryingK'][0][1][0]) 
        for h in FileIO.iterateJsonFromFile(hashtagsAnalayzeLocalityIndexAtKFile%'%s_%s'%timeRange)]
        for h, s in sorted(hashtagsLIScore, key=itemgetter(1)):
            print h, s
    @staticmethod
    def plotDifferenceBetweenSourceAndLocalityLattice(timeRange):
        differenceBetweenLattices = [(h['h'], getHaversineDistance(h['src'][0], h['liAtVaryingK'][0][1][1]))
                                     for h in FileIO.iterateJsonFromFile(hashtagsAnalayzeLocalityIndexAtKFile%'%s_%s'%timeRange)]
        for h, s in sorted(differenceBetweenLattices, key=itemgetter(1)):
            print h, s
def plotHashtagsOnUSMap(timeRange):
    hashtagClasses = HashtagClass.getHashtagClasses()
    classesToPlot = ['technology', 'tv', 'movie', 'sports', 'occupy', 'events', 'republican_debates', 'song_game_releases']
    for cls in classesToPlot:
        points, labels, scores, sources = [], [], [], []
        for h in HashtagObject.iterateHashtagObjects(timeRange, hastagsList=hashtagClasses[cls]):
            score, point = h.getLocalLattice()
            points.append(point), labels.append(h.dict['h']), scores.append(score), sources.append(h.dict['src'])
#            print score, point
        for i, j, k, l in zip(labels, scores, points, sources):
            print i, j, k, l
        print '\n\n\n\n'
#        plotPointsOnUSMap(points, pointLabels=labels)
#        plt.show()
#        exit()

def plotHashtagFlowOnUSMap(sourceLattice, outputFolder):
    def getNodesFromFile(graphFile): return dict([(data['id'], data['links']) for data in FileIO.iterateJsonFromFile(graphFile)])
#    nodes = getNodesFromFile('../data/hashtagSharingProbabilityGraph')
    nodes = getNodesFromFile(hashtagSharingProbabilityGraphFile%(outputFolder,'%s_%s'%timeRange))
    i=0
    for node in nodes:
        sourceLattice = getLocationFromLid(node.replace('_', ' '))
        latticeNodeId = getLatticeLid(sourceLattice, accuracy=ACCURACY)
        outputFileName = hashtagsImagesHastagsSharingProbabilitiesFolder%outputFolder+'%s.png'%latticeNodeId
        FileIO.createDirectoryForFile(outputFileName)
        print i, len(nodes), outputFileName; i+=1
        points, colors = zip(*sorted(nodes[latticeNodeId].iteritems(), key=itemgetter(1)))
        points = [getLocationFromLid(p.replace('_', ' ')) for p in points]
        cm = matplotlib.cm.get_cmap('cool')
        sc = plotPointsOnUSMap(points, c=colors, cmap=cm, lw = 0, alpha=1.0, vmin=0.0, vmax=1.0)
        plotPointsOnUSMap([sourceLattice], c='r', lw=0)
        plt.colorbar(sc)
        plt.savefig(outputFileName)
        plt.clf()
        if i==10: exit()
        
        
        
#class LocationGraphAnalysis():
#    @staticmethod

def getDiGraph(graphFile):
    graph = nx.DiGraph()
    for n in FileIO.iterateJsonFromFile(hashtagSharingProbabilityGraphFile%(outputFolder,'%s_%s'%timeRange)):
        for dest, w in n['links'].iteritems(): graph.add_edge(n['id'], dest, {'w':w})
    return graph


def getValidOccurences(occ, TIME_UNIT_IN_SECONDS = 60*60, MIN_OBSERVATIONS_PER_TIME_UNIT=5): 
    def getValidTimeUnits(occ):
        occurranceDistributionInEpochs = [(k[0], len(list(k[1]))) for k in groupby(sorted([GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) for t in zip(*occ)[1]]))]
        return [t[0] for t in occurranceDistributionInEpochs if t[1]>=MIN_OBSERVATIONS_PER_TIME_UNIT]
    validTimeUnits = getValidTimeUnits(occ)
    return [(p,t) for p,t in occ if GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) in validTimeUnits]


def tempAnalysis(timeRange, outputFolder):
#    points = [getLocationFromLid(n['id'].replace('_', ' ')) for n in FileIO.iterateJsonFromFile(hashtagSharingProbabilityGraphFile%(outputFolder,'%s_%s'%timeRange))]
#    plotPointsOnWorldMap(points)
#    plt.show()

#    graph = getDiGraph(hashtagSharingProbabilityGraphFile%(outputFolder,'%s_%s'%timeRange))
#    plot(graph)

    import datetime
    TIME_UNIT_IN_SECONDS = 60*60
    MIN_OBSERVATIONS_PER_TIME_UNIT = 5
    for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(outputFolder,'%s_%s'%timeRange)):
#        print len(h['oc']), len(getValidOccurences(h['oc']))
        occ = h['oc']
        occurranceDistributionInEpochs = sorted([(k[0], len(list(k[1]))) 
                                                 for k in groupby(sorted(
                                                                          [GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) for t in zip(*occ)[1]])
                                                                  )
                                                 ], key=itemgetter(0))
        occurranceDistributionInEpochs = filter(lambda t: t[1]>=MIN_OBSERVATIONS_PER_TIME_UNIT, occurranceDistributionInEpochs)
        dataX, dataY = zip(*occurranceDistributionInEpochs)
        plt.plot_date(map(datetime.datetime.fromtimestamp, dataX), dataY, '-')
        plt.show()
        exit()

#class PlotsOnMap:
#    TIME_UNIT_IN_SECONDS = 60*60
#    MIN_OBSERVATIONS_PER_TIME_UNIT = 10
#    @staticmethod
#    def getValidTimeUnits(occ):
#        occurranceDistributionInEpochs = [(k[0], len(list(k[1]))) for k in groupby(sorted([GeneralMethods.approximateEpoch(t, PlotsOnMap.TIME_UNIT_IN_SECONDS) for t in zip(*occ)[1]]))]
#        return [t[0] for t in occurranceDistributionInEpochs if t[1]>=PlotsOnMap.MIN_OBSERVATIONS_PER_TIME_UNIT]
#    @staticmethod
#    def getValidOccurences(occ, validTimeUnits): return [(p,t) for p,t in occ if GeneralMethods.approximateEpoch(t, PlotsOnMap.TIME_UNIT_IN_SECONDS) in validTimeUnits]
        
#    @staticmethod
#    def plotHashtagFlowInTime(timeRange, outputFolder):
#        HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS = 1*60*60
#        MIN_OBSERVATIONS_PER_TIME_UNIT = 10
#        def getValidTimeUnits(h):
#            occurranceDistributionInEpochs = [(k[0], len(list(k[1]))) for k in groupby(sorted([GeneralMethods.approximateEpoch(t, HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS) for t in zip(*h['oc'])[1]]))]
#            return [t[0] for t in occurranceDistributionInEpochs if t[1]>=MIN_OBSERVATIONS_PER_TIME_UNIT]
#        for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange)):
#            if h['h'].startswith('occ') or h['h'] in ['ows']:
#                print h['h']
#                validEpochs = getValidTimeUnits(h)
#                observedLattices = {}
#                for lid, t in [(getLatticeLid(l, ACCURACY), t) for l, t in h['oc']]:
#                    if lid not in observedLattices: observedLattices[lid] = t
#                points = sorted([(getLocationFromLid(l.replace('_', ' ')), GeneralMethods.approximateEpoch(t, HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS)) for l, t in observedLattices.iteritems() if GeneralMethods.approximateEpoch(t, HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS) in validEpochs], key=itemgetter(1))
#                points, colors = zip(*points)
#                
#                colors = [(c-colors[0])/HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS for c in colors]
#                cm = matplotlib.cm.get_cmap('autumn')
#                sc = plotPointsOnWorldMap(points, c=colors, cmap=cm, lw = 0, alpha=1.0)
#                plt.colorbar(sc), plt.title(h['h'])
#                plt.show()
#    @staticmethod
#    def plotHashtagFlowInTimeForFirstNLocations(timeRange, outputFolder):
#        for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange)):
#            try:
##                if h['h'].startswith('occ') or h['h'] in ['ows']:
#                for PERCENTAGE_OF_EARLIEST_LOCATIONS in [0.25*i for i in range(1,5)]:
#                    outputFile = hashtagsImagesFlowInTimeForFirstNLocationsFolder%h['h']+'%d_%s.png'%(PERCENTAGE_OF_EARLIEST_LOCATIONS/0.25, PERCENTAGE_OF_EARLIEST_LOCATIONS); FileIO.createDirectoryForFile(outputFile)
#                    print outputFile
#                    validEpochs = PlotsOnMap.getValidTimeUnits(h['oc'])
#                    observedLattices = {}
#                    for lid, t in [(getLatticeLid(l, ACCURACY), t) for l, t in h['oc']]:
#                        if lid not in observedLattices: observedLattices[lid] = t
#                    points = sorted([(getLocationFromLid(l.replace('_', ' ')), t) for l, t in observedLattices.iteritems() if GeneralMethods.approximateEpoch(t, PlotsOnMap.TIME_UNIT_IN_SECONDS) in validEpochs], key=itemgetter(1), reverse=True)
#                    points = points[-1*int(PERCENTAGE_OF_EARLIEST_LOCATIONS*len(points)):]
#                    points, colors = zip(*points)
#                    
#                    colors = [(c-colors[-1])/PlotsOnMap.TIME_UNIT_IN_SECONDS for c in colors]
#                    cm = matplotlib.cm.get_cmap('autumn')
#                    sc = plotPointsOnWorldMap(points, c=colors, cmap=cm, lw = 0, alpha=1.0)
#                    plt.colorbar(sc), plt.title(h['h'])
#    #                plt.show()
#                    plt.savefig(outputFile); plt.clf()
#            except: pass
#            exit()
#
#
#    @staticmethod
#    def plotDistributionGraphs(occurences, validTimeUnits, title, startingEpoch=None):
#        occurences = PlotsOnMap.getValidOccurences(occurences, validTimeUnits)
#        occurancesGroupedByLattice = [(getLocationFromLid(lid.replace('_', ' ')), sorted(zip(*occs)[1])) for lid, occs in groupby(sorted([(getLatticeLid(l, ACCURACY), t) for l, t in occurences], key=itemgetter(0)), key=itemgetter(0))]
#        plt.subplot(211)
#        pointsForNumberOfOccurances, numberOfOccurancesList = zip(*sorted(occurancesGroupedByLattice, key=lambda t: len(t[1])))
#        numberOfOccurancesList = [len(ocs) for ocs in numberOfOccurancesList]
#        cm = matplotlib.cm.get_cmap('cool')
#        sc = plotPointsOnWorldMap(pointsForNumberOfOccurances, c=numberOfOccurancesList, cmap=cm, lw = 0, alpha=1.0)
#        plt.colorbar(sc), plt.title(title), plt.xlabel('Number of mentions')
#        
#        plt.subplot(212)
#        pointsForNumberOfOccurances, occuranceTime = zip(*sorted(occurancesGroupedByLattice, key=lambda t: min(t[1]), reverse=True))
#        occuranceTime=[min(t) for t in occuranceTime]
#        if not startingEpoch: startingEpoch = occuranceTime[-1]
#        occuranceTime=[(t-startingEpoch)/PlotsOnMap.TIME_UNIT_IN_SECONDS for t in occuranceTime]
#        cm = matplotlib.cm.get_cmap('autumn')
#        sc = plotPointsOnWorldMap(pointsForNumberOfOccurances, c=occuranceTime, cmap=cm, lw = 0, alpha=1.0)
#        plt.colorbar(sc), plt.xlabel('Speed of hashtag arrival')
#        return startingEpoch

#    @staticmethod
#    def plotHashtagFlowInTimeForFirstNOccurences(timeRange, outputFolder):
#        numberOfObservedHashtags = 1
#        for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange)):
#            if not os.path.exists(hashtagsImagesFlowInTimeForFirstNOccurrencesFolder%h['h']):
#                validTimeUnits = PlotsOnMap.getValidTimeUnits(h['oc'])
#                for PERCENTAGE_OF_EARLIEST_OCCURENCES in [0.1*i for i in range(1,11)]:
#                    try:
#                        outputFile = hashtagsImagesFlowInTimeForFirstNOccurrencesFolder%h['h']+'%d_%s.png'%(PERCENTAGE_OF_EARLIEST_OCCURENCES/0.1, PERCENTAGE_OF_EARLIEST_OCCURENCES); FileIO.createDirectoryForFile(outputFile)
#                        print numberOfObservedHashtags, outputFile
#                        occurences = h['oc'][:int(PERCENTAGE_OF_EARLIEST_OCCURENCES*len(h['oc']))]
#                        PlotsOnMap.plotDistributionGraphs(occurences, validTimeUnits, '%s - %d'%(h['h'], 100*PERCENTAGE_OF_EARLIEST_OCCURENCES)+'%')
##                        plt.show()
#                        plt.savefig(outputFile); plt.clf()
#                    except: pass
#            numberOfObservedHashtags+=1
#            
#    @staticmethod
#    def plotHashtagFlowInTimeForWindowOfNOccurences(timeRange, outputFolder):
#        numberOfObservedHashtags = 1
#        for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange)):
#            previousIndex, startingEpoch = 0, None
#            if not os.path.exists(hashtagsImagesFlowInTimeForWindowOfNOccurrencesFolder%h['h']):
#                validTimeUnits = PlotsOnMap.getValidTimeUnits(h['oc'])
#                for PERCENTAGE_OF_EARLIEST_OCCURENCES in [0.1*i for i in range(1,11)]:
#                    try:
#                        outputFile = hashtagsImagesFlowInTimeForWindowOfNOccurrencesFolder%h['h']+'%d_%s.png'%(PERCENTAGE_OF_EARLIEST_OCCURENCES/0.1, PERCENTAGE_OF_EARLIEST_OCCURENCES); FileIO.createDirectoryForFile(outputFile)
#                        print numberOfObservedHashtags, outputFile
#                        currentIndex = int(PERCENTAGE_OF_EARLIEST_OCCURENCES*len(h['oc']))
#                        occurences = h['oc'][previousIndex:currentIndex]; previousIndex=currentIndex
#                        startingEpoch = PlotsOnMap.plotDistributionGraphs(occurences, validTimeUnits, '%s - Interval %d'%(h['h'], PERCENTAGE_OF_EARLIEST_OCCURENCES/0.1)+'', startingEpoch)
#    #                    plt.show()
#                        plt.savefig(outputFile); plt.clf()
#                    except: pass
#            numberOfObservedHashtags+=1
##            exit()

#    @staticmethod
#    def ana(timeRange, outputFolder):
#        numberOfObservedHashtags = 1
#        dataX = []
#        for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange)):
#            dataX.append(len(h['oc']))
#            if len(h['oc'])>2000: print numberOfObservedHashtags, len(h['oc']), unicode(h['h']).encode('utf-8'); numberOfObservedHashtags+=1
#        plt.hist(dataX, bins=100)
#        plt.show()
if __name__ == '__main__':
    timeRange = (2,11)
    outputFolder = 'world'
#    plotHashtagFlowOnUSMap([41.046217,-73.652344], outputFolder)

    tempAnalysis(timeRange, outputFolder)
    
#    PlotsOnMap.plotHashtagFlowInTimeForFirstNLocations(timeRange, outputFolder)
#    PlotsOnMap.plotHashtagFlowInTimeForFirstNOccurences(timeRange, outputFolder)
#    PlotsOnMap.plotHashtagFlowInTimeForWindowOfNOccurences(timeRange, outputFolder)
#    PlotsOnMap.ana(timeRange, outputFolder)
    
#    AnalyzeLocalityIndexAtK.LIForOccupy(timeRange)
#    AnalyzeLocalityIndexAtK.rankHashtagsBYLIScore(timeRange)
#    AnalyzeLocalityIndexAtK.plotDifferenceBetweenSourceAndLocalityLattice(timeRange)