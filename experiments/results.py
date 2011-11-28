'''
Created on Nov 24, 2011

@author: kykamath
'''
import sys
sys.path.append('../')
from library.file_io import FileIO
import matplotlib
from settings import hashtagsAnalayzeLocalityIndexAtKFile,\
    hashtagsWithoutEndingWindowFile, hashtagSharingProbabilityGraphFile,\
    hashtagsImagesHastagsSharingProbabilitiesFolder
import matplotlib.pyplot as plt
from operator import itemgetter
from library.geo import getHaversineDistance, plotPointsOnUSMap, getLatticeLid,\
    getLocationFromLid, plotPointsOnWorldMap
from itertools import groupby
from experiments.analysis import HashtagClass, HashtagObject
from experiments.mr_analysis import ACCURACY
from library.classes import GeneralMethods

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
        
class PlotsOnMap:
    @staticmethod
    def plotHashtagFlowInTime(timeRange, outputFolder):
        HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS = 1*60*60
        MIN_OBSERVATIONS_PER_TIME_UNIT = 10
        def getValidEpochs(h):
            occurranceDistributionInEpochs = [(k[0], len(list(k[1]))) for k in groupby(sorted([GeneralMethods.approximateEpoch(t, HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS) for t in zip(*h['oc'])[1]]))]
            return [t[0] for t in occurranceDistributionInEpochs if t[1]>=MIN_OBSERVATIONS_PER_TIME_UNIT]
        for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange)):
            if h['h'].startswith('occ') or h['h'] in ['ows']:
                print h['h']
                validEpochs = getValidEpochs(h)
                observedLattices = {}
                for lid, t in [(getLatticeLid(l, ACCURACY), t) for l, t in h['oc']]:
                    if lid not in observedLattices: observedLattices[lid] = t
                points = sorted([(getLocationFromLid(l.replace('_', ' ')), GeneralMethods.approximateEpoch(t, HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS)) for l, t in observedLattices.iteritems() if GeneralMethods.approximateEpoch(t, HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS) in validEpochs], key=itemgetter(1))
                points, colors = zip(*points)
                
                colors = [(c-colors[0])/HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS for c in colors]
                cm = matplotlib.cm.get_cmap('autumn')
                sc = plotPointsOnWorldMap(points, c=colors, cmap=cm, lw = 0, alpha=1.0)
                plt.colorbar(sc), plt.title(h['h'])
                plt.show()
    @staticmethod
    def plotHashtagFlowInTimeForFirstNLocations(timeRange, outputFolder):
        HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS = 1*60*60
        TIME_UNIT_IN_SECONDS = 60*60
        MIN_OBSERVATIONS_PER_TIME_UNIT = 10
        PERCENTAGE_OF_EARLIEST_LOCATIONS = 0.25
        def getValidEpochs(h):
            occurranceDistributionInEpochs = [(k[0], len(list(k[1]))) for k in groupby(sorted([GeneralMethods.approximateEpoch(t, HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS) for t in zip(*h['oc'])[1]]))]
            return [t[0] for t in occurranceDistributionInEpochs if t[1]>=MIN_OBSERVATIONS_PER_TIME_UNIT]
        for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange)):
            try:
#                if h['h'].startswith('occ') or h['h'] in ['ows']:
                    print h['h']
                    validEpochs = getValidEpochs(h)
                    observedLattices = {}
                    for lid, t in [(getLatticeLid(l, ACCURACY), t) for l, t in h['oc']]:
                        if lid not in observedLattices: observedLattices[lid] = t
                    points = sorted([(getLocationFromLid(l.replace('_', ' ')), t) for l, t in observedLattices.iteritems() if GeneralMethods.approximateEpoch(t, HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS) in validEpochs], key=itemgetter(1), reverse=True)
                    points = points[-1*int(PERCENTAGE_OF_EARLIEST_LOCATIONS*len(points)):]
                    points, colors = zip(*points)
                    
                    colors = [(c-colors[-1])/TIME_UNIT_IN_SECONDS for c in colors]
                    cm = matplotlib.cm.get_cmap('autumn')
                    sc = plotPointsOnWorldMap(points, c=colors, cmap=cm, lw = 0, alpha=1.0)
                    plt.colorbar(sc), plt.title(h['h'])
                    plt.show()
            except: pass

if __name__ == '__main__':
    timeRange = (2,11)
    outputFolder = 'world'
#    plotHashtagFlowOnUSMap([41.046217,-73.652344], outputFolder)
    
    PlotsOnMap.plotHashtagFlowInTimeForFirstNLocations(timeRange, outputFolder)
    
#    AnalyzeLocalityIndexAtK.LIForOccupy(timeRange)
#    AnalyzeLocalityIndexAtK.rankHashtagsBYLIScore(timeRange)
#    AnalyzeLocalityIndexAtK.plotDifferenceBetweenSourceAndLocalityLattice(timeRange)