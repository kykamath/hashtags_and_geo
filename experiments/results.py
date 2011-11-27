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
    getLocationFromLid
from itertools import groupby
from experiments.analysis import HashtagClass, HashtagObject
from experiments.mr_analysis import ACCURACY

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
    for node in nodes:
        sourceLattice = getLocationFromLid(node.replace('_', ' '))
        latticeNodeId = getLatticeLid(sourceLattice, accuracy=ACCURACY)
        outputFileName = hashtagsImagesHastagsSharingProbabilitiesFolder%outputFolder+'%s.png'%latticeNodeId
        FileIO.createDirectoryForFile(outputFileName)
        print outputFileName
    #    points, colors = zip(*sorted(nodes[latticeNodeId].iteritems(), key=itemgetter(1)))
    #    points = [getLocationFromLid(p.replace('_', ' ')) for p in points]
    #    cm = matplotlib.cm.get_cmap('gist_yarg')
    #    sc = plotPointsOnUSMap(points, c=colors, cmap=cm, lw = 0, alpha=1.0)
    #    plotPointsOnUSMap([sourceLattice], c='r', lw=0)
    #    plt.colorbar(sc)
    #    plt.savefig(outputFileName)
    

if __name__ == '__main__':
    timeRange = (2,11)
    outputFolder = '/'
    plotHashtagFlowOnUSMap([41.046217,-73.652344], outputFolder)
    
#    AnalyzeLocalityIndexAtK.LIForOccupy(timeRange)
#    AnalyzeLocalityIndexAtK.rankHashtagsBYLIScore(timeRange)
#    AnalyzeLocalityIndexAtK.plotDifferenceBetweenSourceAndLocalityLattice(timeRange)