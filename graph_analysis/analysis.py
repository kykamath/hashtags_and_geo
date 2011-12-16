'''
Created on Dec 15, 2011

@author: kykamath
'''
import sys, datetime
import math
from networkx.generators.random_graphs import erdos_renyi_graph,\
    fast_gnp_random_graph
sys.path.append('../')
from library.geo import getLocationFromLid, plotPointsOnWorldMap,\
    plotPointsOnUSMap, isWithinBoundingBox
from library.classes import GeneralMethods, timeit
from library.mrjobwrapper import runMRJob
from itertools import groupby
from library.file_io import FileIO
from graph_analysis.mr_modules import MRGraph, TIME_UNIT_IN_SECONDS, updateNode,\
    updateEdge
from graph_analysis.settings import hdfsInputFolder, epochGraphsFile,\
    hashtagsFile, us_boundary, runningTimesFolder, randomGraphsFolder
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

#def combine(g1, g2):
#    for u,v,data in g2.edges(data=True): updateNode(g1, u, g2.node[u]['w']), updateNode(g1, v, g2.node[v]['w']), updateEdge(g1, u, v, data['w'])
#    return g1

def combineGraphList(graphs):
    graph = nx.Graph()
    def addToG(g):
        nodesUpdated = set()
        for u,v,data in g.edges(data=True): 
            if u not in nodesUpdated: updateNode(graph, u, g.node[u]['w']), nodesUpdated.add(u)
            if v not in nodesUpdated: updateNode(graph, v, g.node[v]['w']), nodesUpdated.add(v)
            updateEdge(graph, u, v, data['w'])
    for g in graphs: addToG(g)
    return graph

class RandomGraphGenerator:
    fast_gnp_random_graph = 'fast_gnp_random_graph'
    erdos_renyi_graph='erdos_renyi_graph'
    @staticmethod
    def fastGnp(n,p=0.3):
        graphsToReturn = []
        for i in range(100): 
            print RandomGraphGenerator.fast_gnp_random_graph, n, i
            graphsToReturn.append([i*TIME_UNIT_IN_SECONDS, my_nx.getDictForGraph(fast_gnp_random_graph(n,p))])
        return graphsToReturn
    @staticmethod
    def erdosRenyi(n,p=0.3):
        graphsToReturn = []
        for i in range(100): 
            print RandomGraphGenerator.erdos_renyi_graph, n, i
            graphsToReturn.append([i*TIME_UNIT_IN_SECONDS, my_nx.getDictForGraph(erdos_renyi_graph(n,p))])
        return graphsToReturn
    @staticmethod
    def run():
        for graphType, method in [(RandomGraphGenerator.fast_gnp_random_graph, RandomGraphGenerator.fastGnp), \
                                  (RandomGraphGenerator.erdos_renyi_graph, RandomGraphGenerator.erdosRenyi)]:
            for i in range(1, 11): FileIO.writeToFileAsJson({'n': 100*i, 'graphs': method(100*i)}, randomGraphsFolder%graphType)

class LocationGraphs:
    @staticmethod
    def getLogarithmicGraphId(startingGraphId, graphId): return ((int(graphId)-startingGraphId)/TIME_UNIT_IN_SECONDS)+1
    @staticmethod
    def getGraphId(startingGraphId, logarithmicGraphId): return startingGraphId+((logarithmicGraphId-1)*TIME_UNIT_IN_SECONDS)
    @staticmethod
    @timeit
    def combineLocationGraphs(graphMap, startingGraphId, startingTime, intervalInSeconds, linear=True, **kwargs):
        if intervalInSeconds%TIME_UNIT_IN_SECONDS==0 and int(intervalInSeconds/TIME_UNIT_IN_SECONDS)!=0: numberOfGraphs = int(intervalInSeconds/TIME_UNIT_IN_SECONDS)
        else: numberOfGraphs = int(intervalInSeconds/TIME_UNIT_IN_SECONDS)+1
        graphId = GeneralMethods.approximateEpoch(GeneralMethods.getEpochFromDateTimeObject(startingTime), TIME_UNIT_IN_SECONDS)
        currentLogarithmicId = LocationGraphs.getLogarithmicGraphId(startingGraphId, graphId)
        currentCollectedGraphs = 0
        graphIdsToCombine = []
        while currentCollectedGraphs!=numberOfGraphs and currentLogarithmicId>0:
            numberOfGraphsToCollect = 2**int(math.log(numberOfGraphs-currentCollectedGraphs,2))
            if not linear and currentLogarithmicId%2==0: 
                indices = [1]+map(lambda j: 2**j, filter(lambda j: currentLogarithmicId%(2**j)==0, range(1, int(math.log(currentLogarithmicId+1,2))+1)))
                if max(indices)>numberOfGraphsToCollect and numberOfGraphsToCollect in indices: index = numberOfGraphsToCollect
                else: index = max(indices)
            else: index=1
            logGraphId = '%s_%s'%(LocationGraphs.getGraphId(startingGraphId, currentLogarithmicId), index)
            if logGraphId in graphMap: graphIdsToCombine.append(logGraphId)
            currentLogarithmicId-=index
            currentCollectedGraphs+=index
        graphIdsToCombine = sorted(graphIdsToCombine, key=lambda id:int(id.split('_')[1]), reverse=True)
        graphsToCombine = [graphMap[id] for id in graphIdsToCombine]
        return combineGraphList(graphsToCombine)
    @staticmethod
    def updateLogarithmicGraphs(graphMap):
        print 'Building logarithmic graphs... ',
        startingGraphId = sorted(graphMap.keys())[0]
        for graphId in sorted(graphMap.keys()):
            i = LocationGraphs.getLogarithmicGraphId(startingGraphId, graphId)
            if i%2==0: 
                indices = map(lambda j: 2**j, filter(lambda j: i%(2**j)==0, range(1, int(math.log(i+1,2))+1)))
                for graphIdsToCombine in [map(lambda j: graphId-j*TIME_UNIT_IN_SECONDS, range(index)) for index in indices]:
                    graphsToCombine = [graphMap[j] for j in graphIdsToCombine if j in graphMap]
                    graphMap['%s_%s'%(graphId, len(graphIdsToCombine))] = combineGraphList(graphsToCombine)
            graphMap['%s_%s'%(graphId, 1)] = graphMap[graphId]; del graphMap[graphId]
        print 'Completed!!'
        return startingGraphId
    @staticmethod
    def runningTimeAnalysis(graphs, graphType, numberOfPoints=50):
        def getRunningTime(graphs, linear):
            graphMap = dict(graphs)
            startingGraphId, endingGraphId = min(graphMap.keys()), max(graphMap.keys())
            timeDifference = endingGraphId-startingGraphId
            LocationGraphs.updateLogarithmicGraphs(graphMap)
            dataToReturn = []
            for j, intervalInSeconds in enumerate(range(0, timeDifference, int(timeDifference/numberOfPoints))):
                graph, runningTime = LocationGraphs.combineLocationGraphs(graphMap, startingGraphId, datetime.datetime.fromtimestamp(endingGraphId+1), intervalInSeconds, linear=linear, returnTimeDifferenceOnly=True)
                print graphType, linear, j, runningTime
                dataToReturn.append([intervalInSeconds, runningTime])
            return dataToReturn
#        getRunningTime(graphs, True)
#        exit()
        graphFile = runningTimesFolder%graphType
        print graphFile
        GeneralMethods.runCommand(graphFile)
        for linear in [True, False]: FileIO.writeToFileAsJson({'linear': linear, 'running_time': getRunningTime(graphs, linear)}, graphFile)
    @staticmethod
    def run():
        timeRange, dataType, area = (5,11), 'world', 'world'
        graphs = getGraphs(area, timeRange)
        LocationGraphs.runningTimeAnalysis(graphs, 'location')
    
    
def getGraphs(area, timeRange): return sorted([(d['ep'], my_nx.getGraphFromDict(d['graph']))for d in FileIO.iterateJsonFromFile(epochGraphsFile%(area, '%s_%s'%timeRange))])
def temp_analysis():
    graphMap = dict(getGraphs(area, timeRange))
    startingGraphId = LocationGraphs.updateLogarithmicGraphs(graphMap)
    startingTime, intervalInSeconds = datetime.datetime(2011,11,15,7,7,30), 60*TIME_UNIT_IN_SECONDS
    graph, runningTime = LocationGraphs.combineLocationGraphs(graphMap, startingGraphId, startingTime, intervalInSeconds, linear=False, returnTimeDifferenceOnly=True)
    print graph.number_of_nodes(), runningTime
#    print clusterUsingAffinityPropagation(graph)

#    for i, (ep, graph) in enumerate(getGraphs(area, timeRange)):
#        print datetime.datetime.fromtimestamp(ep)
#        plotLocationClustersOnMap(graph)
        
        

def getInputFiles(months, folderType='/'): return [hdfsInputFolder+folderType+'/'+str(m) for m in months] 
def mr_task(timeRange, dataType, area):
#    runMRJob(MRGraph, hashtagsFile%(area, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), dataType), jobconf={'mapred.reduce.tasks':160})
    runMRJob(MRGraph, epochGraphsFile%(area, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), dataType), jobconf={'mapred.reduce.tasks':160})   

if __name__ == '__main__':
#    timeRange, dataType, area = (5,6), 'world', 'us'
    timeRange, dataType, area = (5,11), 'world', 'world'
    
#    mr_task(timeRange, dataType, area)
#    temp_analysis()
#    LocationGraphs.run()
    RandomGraphGenerator.run()