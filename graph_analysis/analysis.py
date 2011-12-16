'''
Created on Dec 15, 2011

@author: kykamath
'''
import sys, datetime
import math
sys.path.append('../')
from library.geo import getLocationFromLid, plotPointsOnWorldMap,\
    plotPointsOnUSMap, isWithinBoundingBox
from library.classes import GeneralMethods
from library.mrjobwrapper import runMRJob
from itertools import groupby
from library.file_io import FileIO
from graph_analysis.mr_modules import MRGraph, TIME_UNIT_IN_SECONDS, updateNode,\
    updateEdge
from graph_analysis.settings import hdfsInputFolder, epochGraphsFile,\
    hashtagsFile, us_boundary
import networkx as nx
from operator import itemgetter
from library.graphs import Networkx as my_nx
from library.graphs import clusterUsingAffinityPropagation
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#    n_clusters_ = len(cluster_centers_indices)


def plotLocationGraphOnMap(graph):
    points = map(lambda  lid:getLocationFromLid(lid.replace('_', ' ')), graph.nodes())
    _, m =plotPointsOnUSMap(points, s=10, lw=0, c='m', returnBaseMapObject=True)
    print graph.number_of_edges(), graph.number_of_nodes()
    totalEdgeWeight = max([d['w'] for _,_,d in graph.edges(data=True)])+0.0
    for u, v, data in graph.edges(data=True):
        u, v, w = getLocationFromLid(u.replace('_', ' ')), getLocationFromLid(v.replace('_', ' ')), data['w']
        m.drawgreatcircle(u[1],u[0],v[1],v[0],color=cm.Purples(w/totalEdgeWeight), alpha=0.5)
    plt.show()

def plotLocationClustersOnMap(graph):
    noOfClusters, clusters = clusterUsingAffinityPropagation(graph)
    nodeToClusterIdMap = dict(clusters)
    colorMap = dict([(i, GeneralMethods.getRandomColor()) for i in range(noOfClusters)])
    clusters = [(c, list(l)) for c, l in groupby(sorted(clusters, key=itemgetter(1)), key=itemgetter(1))]
    points, colors = zip(*map(lambda  l: (getLocationFromLid(l.replace('_', ' ')), colorMap[nodeToClusterIdMap[l]]), graph.nodes()))
    _, m =plotPointsOnUSMap(points, s=30, lw=0, c=colors, returnBaseMapObject=True)
    for u, v, data in graph.edges(data=True):
        if nodeToClusterIdMap[u]==nodeToClusterIdMap[v]:
            color, u, v, w = colorMap[nodeToClusterIdMap[u]], getLocationFromLid(u.replace('_', ' ')), getLocationFromLid(v.replace('_', ' ')), data['w']
            m.drawgreatcircle(u[1],u[0],v[1],v[0],color=color, alpha=0.5)
    plt.show()

def combine(g1, g2):
    for u,v,data in g2.edges(data=True): updateNode(g1, u, g2.node[u]['w']), updateNode(g1, v, g2.node[v]['w']), updateEdge(g1, u, v, data['w'])
    return g1

def linearCombineGraphs(graphMap, startingTime, intervalInSeconds):
    if intervalInSeconds%TIME_UNIT_IN_SECONDS==0: numberOfGraphs = int(intervalInSeconds/TIME_UNIT_IN_SECONDS)
    else: numberOfGraphs = int(intervalInSeconds/TIME_UNIT_IN_SECONDS)+1
    graphId = GeneralMethods.approximateEpoch(GeneralMethods.getEpochFromDateTimeObject(startingTime), TIME_UNIT_IN_SECONDS)
    graphIdsToCombine = map(lambda i: graphId-TIME_UNIT_IN_SECONDS*i, range(numberOfGraphs))
    graphsToCombine = [graphMap[id] for id in graphIdsToCombine if id in graphMap]
    return reduce(combine,graphsToCombine[1:],graphsToCombine[0])

def getLogarithmicGraphId(startingGraphId, graphId): return (graphId-startingGraphId)/TIME_UNIT_IN_SECONDS
def logarithmicCombineGraphs(graphMap, startingGraphId, startingTime, intervalInSeconds):
    print startingGraphId, startingTime, intervalInSeconds
    if intervalInSeconds%TIME_UNIT_IN_SECONDS==0: numberOfGraphs = int(intervalInSeconds/TIME_UNIT_IN_SECONDS)
    else: numberOfGraphs = int(intervalInSeconds/TIME_UNIT_IN_SECONDS)+1
    graphId = GeneralMethods.approximateEpoch(GeneralMethods.getEpochFromDateTimeObject(startingTime), TIME_UNIT_IN_SECONDS)
    print getLogarithmicGraphId(startingGraphId, graphId)
def updateLogarithmicGraphs(graphMap):
    print 'Building logarithmic graphs... ',
    startingGraphId = sorted(graphMap.keys())[0]
    for id in sorted(graphMap.keys()):
        i = getLogarithmicGraphId(startingGraphId, id)+1
        if i%2==0: 
            indices = map(lambda j: j*2, filter(lambda j: i%(2**j)==0, range(1, int(math.log(i+1,2))+1)))
            for graphIdsToCombine in [map(lambda j: id-j*TIME_UNIT_IN_SECONDS, range(index)) for index in indices]:
                graphsToCombine = [graphMap[j] for j in graphIdsToCombine if j in graphMap]
                graphMap['%s_%s'%(id, len(graphIdsToCombine))] = reduce(combine,graphsToCombine[1:],graphsToCombine[0])
    print 'Completed!!'
    return startingGraphId
    
def getGraphs(area, timeRange): return sorted([(d['ep'], my_nx.getGraphFromDict(d['graph']))for d in FileIO.iterateJsonFromFile(epochGraphsFile%(area, '%s_%s'%timeRange))])
def temp_analysis():
    graphMap = dict(getGraphs(area, timeRange))
    print len(graphMap)
#    startingGraphId = updateLogarithmicGraphs(graphMap)
    startingGraphId = sorted(graphMap)[0]
    print len(graphMap)
    startingTime, intervalInSeconds = datetime.datetime(2011,5,5,6,7,30), 24*TIME_UNIT_IN_SECONDS
#    graph = linearCombineGraphs(graphMap, startingTime, intervalInSeconds)
    graph = logarithmicCombineGraphs(graphMap, startingGraphId, startingTime, intervalInSeconds)

#    print clusterUsingAffinityPropagation(graph)

#    for i, (ep, graph) in enumerate(getGraphs(area, timeRange)):
#        print datetime.datetime.fromtimestamp(ep)
#        plotLocationClustersOnMap(graph)
        
        

def getInputFiles(months, folderType='/'): return [hdfsInputFolder+folderType+'/'+str(m) for m in months] 
def mr_task(timeRange, dataType, area):
#    runMRJob(MRGraph, hashtagsFile%(area, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), dataType), jobconf={'mapred.reduce.tasks':160})
    runMRJob(MRGraph, epochGraphsFile%(area, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), dataType), jobconf={'mapred.reduce.tasks':160})   

if __name__ == '__main__':
    timeRange, dataType, area = (5,6), 'world', 'us'
#    timeRange, dataType, area = (5,6), 'world', 'world'
    
#    mr_task(timeRange, dataType, area)
    temp_analysis()