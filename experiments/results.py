'''
Created on Nov 24, 2011

@author: kykamath
'''
import sys, os
import datetime
from library.plotting import smooth, CurveFit
from library.kml import KML
sys.path.append('../')
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
    hashtagsImagesLocationInfluencersFolder, hashtagsImagesGraphAnalysisFolder
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import fft, array
from collections import defaultdict
from library.graphs import plot, clusterUsingMCLClustering
import matplotlib.pyplot as plt
from operator import itemgetter
from library.geo import getHaversineDistance, plotPointsOnUSMap, getLatticeLid,\
    getLocationFromLid, plotPointsOnWorldMap
from itertools import groupby
from experiments.analysis import HashtagClass, HashtagObject
from experiments.mr_analysis import ACCURACY, getOccurranceDistributionInEpochs,\
    TIME_UNIT_IN_SECONDS, getOccuranesInHighestActiveRegion
from library.classes import GeneralMethods
import networkx as nx

#class AnalyzeLocalityIndexAtK:
#    @staticmethod
#    def LIForOccupy(timeRange):
#        imagesFolder = '/tmp/images/'
#        occupyList = ['occupywallst', 'occupyoakland', 'occupydc', 'occupysf']
#        FileIO.createDirectoryForFile(imagesFolder+'dsf')
#        for h in FileIO.iterateJsonFromFile(hashtagsAnalayzeLocalityIndexAtKFile%'%s_%s'%timeRange):
#    #        if h['h'].startswith('occupy') or h['h']=='ows':
#            if h['h'] in occupyList:
#                print h['h']
#                dataX, dataY = zip(*[(k, v[0])for k, v in h['liAtVaryingK']])
#                plt.plot(dataX, dataY, label=h['h'])
#    #    plt.title(h['h'])
#        plt.ylim(ymax=4000)
#        plt.legend(loc=2)
#        plt.show()
#    @staticmethod
#    def rankHashtagsBYLIScore(timeRange):
#        hashtagsLIScore = [(h['h'], h['liAtVaryingK'][0][1][0]) 
#        for h in FileIO.iterateJsonFromFile(hashtagsAnalayzeLocalityIndexAtKFile%'%s_%s'%timeRange)]
#        for h, s in sorted(hashtagsLIScore, key=itemgetter(1)):
#            print h, s
#    @staticmethod
#    def plotDifferenceBetweenSourceAndLocalityLattice(timeRange):
#        differenceBetweenLattices = [(h['h'], getHaversineDistance(h['src'][0], h['liAtVaryingK'][0][1][1]))
#                                     for h in FileIO.iterateJsonFromFile(hashtagsAnalayzeLocalityIndexAtKFile%'%s_%s'%timeRange)]
#        for h, s in sorted(differenceBetweenLattices, key=itemgetter(1)):
#            print h, s
#def plotHashtagsOnUSMap(timeRange):
#    hashtagClasses = HashtagClass.getHashtagClasses()
#    classesToPlot = ['technology', 'tv', 'movie', 'sports', 'occupy', 'events', 'republican_debates', 'song_game_releases']
#    for cls in classesToPlot:
#        points, labels, scores, sources = [], [], [], []
#        for h in HashtagObject.iterateHashtagObjects(timeRange, hastagsList=hashtagClasses[cls]):
#            score, point = h.getLocalLattice()
#            points.append(point), labels.append(h.dict['h']), scores.append(score), sources.append(h.dict['src'])
##            print score, point
#        for i, j, k, l in zip(labels, scores, points, sources):
#            print i, j, k, l
#        print '\n\n\n\n'
#        plotPointsOnUSMap(points, pointLabels=labels)
#        plt.show()
#        exit()

#def plotHashtagFlowOnUSMap(sourceLattice, outputFolder):
#    def getNodesFromFile(graphFile): return dict([(data['id'], data['links']) for data in FileIO.iterateJsonFromFile(graphFile)])
##    nodes = getNodesFromFile('../data/hashtagSharingProbabilityGraph')
#    nodes = getNodesFromFile(hashtagSharingProbabilityGraphFile%(outputFolder,'%s_%s'%timeRange))
#    i=0
#    for node in nodes:
#        sourceLattice = getLocationFromLid(node.replace('_', ' '))
#        latticeNodeId = getLatticeLid(sourceLattice, accuracy=ACCURACY)
#        outputFileName = hashtagsImagesHastagsSharingProbabilitiesFolder%outputFolder+'%s.png'%latticeNodeId
#        FileIO.createDirectoryForFile(outputFileName)
#        print i, len(nodes), outputFileName; i+=1
#        points, colors = zip(*sorted(nodes[latticeNodeId].iteritems(), key=itemgetter(1)))
#        points = [getLocationFromLid(p.replace('_', ' ')) for p in points]
#        cm = matplotlib.cm.get_cmap('cool')
#        sc = plotPointsOnUSMap(points, c=colors, cmap=cm, lw = 0, alpha=1.0, vmin=0.0, vmax=1.0)
#        plotPointsOnUSMap([sourceLattice], c='r', lw=0)
#        plt.colorbar(sc)
#        plt.savefig(outputFileName)
#        plt.clf()
#        if i==10: exit()

#def plotNodeObject(timeRange, outputFolder):
#    for nodeObject in FileIO.iterateJsonFromFile(hashtagLocationTemporalClosenessGraphFile%(outputFolder, '%s_%s'%timeRange)): 
#        ax = plt.subplot(111)
#        cm = matplotlib.cm.get_cmap('cool')
#        points, colors = zip(*sorted([(getLocationFromLid(k.replace('_', ' ')), v)for k, v in nodeObject['links'].iteritems() if v>=0.4], key=itemgetter(1)))
#        sc = plotPointsOnWorldMap(points, c=colors, cmap=cm, lw=0)
#        plotPointsOnWorldMap([getLocationFromLid(nodeObject['id'].replace('_', ' '))], c='k', s=20, lw=0)
#        plt.xlabel('Measure of closeness'), plt.title(nodeObject['id'].replace('_', ' '))
#        divider = make_axes_locatable(ax)
#        cax = divider.append_axes("right", size="5%", pad=0.05)
#        plt.colorbar(sc, cax=cax)
#        plt.show()
#            plt.savefig(outputFile); plt.clf()

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
        
def plotGraphs(timeRange, outputFolder):
    sharingProbabilityId = 'sharing_probability'
    temporalClosenessId = 'temporal_closeness'
    def plotPoints(nodeObject, xlabel):
        cm = matplotlib.cm.get_cmap('cool')
        points, colors = zip(*sorted([(getLocationFromLid(k.replace('_', ' ')), v)for k, v in nodeObject['links'].iteritems()], key=itemgetter(1)))
        sc = plotPointsOnWorldMap(points, c=colors, cmap=cm, lw=0)
        plotPointsOnWorldMap([getLocationFromLid(nodeObject['id'].replace('_', ' '))], c='k', s=20, lw=0)
        plt.xlabel(xlabel), plt.colorbar(sc)
    def plotLocationObject(locationObject):
        point = getLocationFromLid(locationObject['id'].replace('_', ' '))
        outputFile = hashtagsImagesLocationClosenessFolder+'%s.png'%getLatticeLid([point[1], point[0]], ACCURACY); FileIO.createDirectoryForFile(outputFile)
        if not os.path.exists(outputFile):
            print outputFile
            plt.subplot(211)
            plt.title(locationObject['id'].replace('_', ' '))
            plotPoints(locationObject['graphs'][sharingProbabilityId], xlabel = 'Hashtag sharing probability')
            plt.subplot(212)
            plotPoints(locationObject['graphs'][temporalClosenessId], xlabel = 'Temporal closeness')
#            plt.show()
            plt.savefig(outputFile); plt.clf()
    locationsMap = defaultdict(dict)
    for node in FileIO.iterateJsonFromFile(hashtagSharingProbabilityGraphFile%(outputFolder, '%s_%s'%timeRange)): 
        if 'graphs' not in locationsMap[node['id']]: locationsMap[node['id']] = {'id': node['id'], 'graphs': {}}
        locationsMap[node['id']]['graphs'][sharingProbabilityId] = node
    for node in FileIO.iterateJsonFromFile(hashtagLocationTemporalClosenessGraphFile%(outputFolder, '%s_%s'%timeRange)): 
        if 'graphs' not in locationsMap[node['id']]: locationsMap[node['id']] = {'id': node['id'], 'graphs': {}}
        locationsMap[node['id']]['graphs'][temporalClosenessId] = node
    for l in locationsMap:
        if len(locationsMap[l]['graphs'])==2:
            plotLocationObject(locationsMap[l])

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
        
        outputFile = hashtagsImagesTimeSeriesAnalysisFolder+'%s.png'%(hashtagObject['h']); FileIO.createDirectoryForFile(outputFile)
        print unicode(outputFile).encode('utf-8')
        
        timeUnits, timeSeries = getDataToPlot(hashtagObject['oc'])
        timeUnitsForActiveRegion, timeSeriesForActiveRegion = getDataToPlot(getOccuranesInHighestActiveRegion(hashtagObject))
        
    #    ax=plt.subplot(311)
    #    plt.plot_date(map(datetime.datetime.fromtimestamp, timeUnits), timeSeries, '-')
    #    plt.setp(ax.get_xticklabels(), rotation=30, fontsize=10)
        ax=plt.subplot(211)
        plt.plot_date(map(datetime.datetime.fromtimestamp, timeUnits), timeSeries, '-')
        plt.plot_date(map(datetime.datetime.fromtimestamp, timeUnitsForActiveRegion), timeSeriesForActiveRegion, 'o', c='r')
        plt.setp(ax.get_xticklabels(), rotation=30, fontsize=10)
        plt.title(hashtagObject['h'])
        ax=plt.subplot(212)
        plt.plot_date(map(datetime.datetime.fromtimestamp, timeUnitsForActiveRegion), timeSeriesForActiveRegion, '-')
        plt.setp(ax.get_xticklabels(), rotation=30, fontsize=10)
#        plt.show()
        plt.savefig(outputFile); plt.clf()
    counter=1
    for object in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange)):
#        if object['h']=='beatohio':
        print counter; counter+=1
        plotTimeSeries(object)

#def getLocationsInfluencGraph(timeRange, outputFolder):
#    graph = nx.DiGraph()
#    counter=1
#    for n in FileIO.iterateJsonFromFile(hashtagLocationInAndOutTemporalClosenessGraphFile%(outputFolder,'%s_%s'%timeRange)):
##        print 'Loading', counter, n['id']; counter+=1
#        for source, w in n['in_link'].iteritems(): 
#            if not graph.has_edge(source, n['id']): graph.add_edge(source, n['id'], {'w':w})
#        for dest, w in n['out_link'].iteritems(): 
#            if not graph.has_edge(n['id'], dest): graph.add_edge(n['id'], dest, {'w':w})
#    
#    return graph
class GraphAnalysis:
    hastagSharingId = 'hashtag_sharing'
    temporalClosenessId = 'temporal_closeness'
    @staticmethod
    def loadGraph(graphFile, timeRange, outputFolder):
        graph = nx.Graph()
        for n in FileIO.iterateJsonFromFile(graphFile%(outputFolder,'%s_%s'%timeRange)):
            for source, w in n['links'].iteritems():  graph.add_edge(source, n['id'], {'w':w})
        return graph
    @staticmethod
    def plotConnectedComponents(timeRange, outputFolder):
        def plotFor(id, graphFile, PERCENTAGE_OF_EDGES):
            graph = GraphAnalysis.loadGraph(graphFile, timeRange, outputFolder)
            plt.hist([graph[u][v]['w'][1] for u,v in graph.edges()], 100)
            plt.show()
            exit()
#            for u,v in graph.edges()[:]: print graph[u][v]    
#            exit()
            plot(graph)
            graph.show()
            exit()
            print len(clusterUsingMCLClustering(graph, inflation=7.0))
            exit()
#            plt.hist([data['w'][1] for u,v, data in graph.edges(data=True)])
#            plt.show()
#            exit()
            imagesOutputFolder = hashtagsImagesGraphAnalysisFolder+'%s/connected_components/'%id
            GeneralMethods.runCommand('rm -rf %s'%imagesOutputFolder)
#            edgesToRemove = sorted([(u,v,data['w']) for u,v,data in graph.edges(data=True)], key=lambda t: t[2][1])[:-int(PERCENTAGE_OF_EDGES*graph.number_of_edges())]
            edgesToRemove = [(u,v,data['w']) for u,v,data in graph.edges(data=True) if data['w'][1]>0.25]
            print graph.number_of_edges(), int(PERCENTAGE_OF_EDGES*graph.number_of_edges()), len(edgesToRemove)
            for u,v,_ in edgesToRemove: graph.remove_edge(u, v)
            print graph.number_of_edges()
#            components = [c for c in nx.connected_components(graph) if len(c)>2]
            print len(components)
            i = 0
            for component in components:
                if len(component)>2:
                    outputFile=imagesOutputFolder+'%s.kml'%i;i+=1;FileIO.createDirectoryForFile(outputFile)
                    print outputFile
                    points = [getLocationFromLid(c.replace('_', ' ')) for c in component]
                    KML.drawKMLsForPoints(points, outputFile, color=GeneralMethods.getRandomColor())
#                    plot(graph.subgraph(component))
#                    for e in graph.subgraph(component).edges(data=True):
#                        print e
#                    exit()
#                    print points
#                    plotPointsOnWorldMap(points, c='m', lw=0)
#                    plotPointsOnUSMap(points, c='m', lw=0)
#                    plt.show()
#                    plt.savefig(outputFile); plt.clf()
        plotFor(GraphAnalysis.hastagSharingId, hashtagSharingProbabilityGraphFile, 0.1)
#        plotFor(GraphAnalysis.temporalClosenessId, hashtagLocationTemporalClosenessGraphFile, 0.1)
    @staticmethod
    def me(): pass    
    

#    plt.hist([e[2]['w'] for e in graph.edges(data=True)], bins=10)
#    plt.show()

#def getOccurencesFilteredByDistributionInTimeUnits(occ, TIME_UNIT_IN_SECONDS = 60*60, MIN_OBSERVATIONS_PER_TIME_UNIT=5): 
#    def getValidTimeUnits(occ):
#        occurranceDistributionInEpochs = [(k[0], len(list(k[1]))) for k in groupby(sorted([GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) for t in zip(*occ)[1]]))]
#        return [t[0] for t in occurranceDistributionInEpochs if t[1]>=MIN_OBSERVATIONS_PER_TIME_UNIT]
#    validTimeUnits = getValidTimeUnits(occ)
#    return [(p,t) for p,t in occ if GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) in validTimeUnits]
    
if __name__ == '__main__':
    timeRange = (2,11)
#    outputFolder = 'world'
    outputFolder='ny'
#    outputFolder='/'
#    plotHashtagFlowOnUSMap([41.046217,-73.652344], outputFolder)

#    tempAnalysis(timeRange, outputFolder)
#    plotTimeSeriesWithHighestActiveRegion(timeRange, outputFolder)
#    plotGraphs(timeRange, outputFolder)
#    plotNodeObject(timeRange, outputFolder)
#    plotHashtagsInOutGraphs(timeRange, outputFolder)

    GraphAnalysis.plotConnectedComponents(timeRange, outputFolder)

    
#    AnalyzeLocalityIndexAtK.LIForOccupy(timeRange)
#    AnalyzeLocalityIndexAtK.rankHashtagsBYLIScore(timeRange)
#    AnalyzeLocalityIndexAtK.plotDifferenceBetweenSourceAndLocalityLattice(timeRange)