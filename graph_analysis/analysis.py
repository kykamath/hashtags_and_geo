'''
Created on Dec 15, 2011

@author: kykamath
'''
import sys, datetime
import math
from networkx.generators.random_graphs import erdos_renyi_graph,\
    fast_gnp_random_graph, newman_watts_strogatz_graph, powerlaw_cluster_graph
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
    hashtagsFile, us_boundary, runningTimesFolder, randomGraphsFolder,\
    tempEpochGraphsFile
import networkx as nx
from operator import itemgetter
from library.graphs import Networkx as my_nx
from library.graphs import clusterUsingAffinityPropagation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import defaultdict
from library.plotting import splineSmooth

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

#def combineGraphList(graphs):
##    graph = nx.Graph()
#    graph = graphs[0].copy()
#    def addToG(g):
#        nodesUpdated = set()
#        for u,v,data in g.edges(data=True): 
#            if u not in nodesUpdated: updateNode(graph, u, g.node[u]['w']), nodesUpdated.add(u)
#            if v not in nodesUpdated: updateNode(graph, v, g.node[v]['w']), nodesUpdated.add(v)
#            updateEdge(graph, u, v, data['w'])
#    for g in graphs[1:]: addToG(g)
#    return graph

#def combineGraphList(graphs):
#    graph = nx.Graph()
#    edgesCount, numberOfGraphs = defaultdict(int), float(len(graphs))
#    for g in graphs:
#        for u, v in g.edges(): edgesCount['%s:ilab:%s'%(tuple(sorted([u,v])))]+=1
#    for e in edgesCount:
#        if edgesCount[e]/numberOfGraphs>=0.5:
#            u,v = e.split(':ilab:')
#            graph.add_edge(u, v)
#    return graph

def combineGraphList(graphs, edgesToKeep=0.25):
#    graph = nx.Graph()
    graph = graphs[0].copy()
    def addToG(g):
        nodesUpdated = set()
        for u,v,data in g.edges(data=True): 
            if u not in nodesUpdated: updateNode(graph, u, g.node[u]['w']), nodesUpdated.add(u)
            if v not in nodesUpdated: updateNode(graph, v, g.node[v]['w']), nodesUpdated.add(v)
            updateEdge(graph, u, v, data['w'])
    for g in graphs[1:]: addToG(g)
    edgesToRemove = sorted([(u,v, data['w']) for u,v,data in graph.edges(data=True)], key=itemgetter(2))[:int(graph.number_of_edges()*(1-edgesToKeep))]
    for u,v,_ in edgesToRemove: graph.remove_edge(u,v)
    return graph

class RandomGraphGenerator:
    fast_gnp_random_graph = 'fast_gnp_random_graph'
    erdos_renyi_graph='erdos_renyi_graph'
    newman_watts_strogatz_graph='newman_watts_strogatz_graph'
    powerlaw_cluster_graph='powerlaw_cluster_graph'
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
    def nWS(n,k=3,p=0.3):
        graphsToReturn = []
        for i in range(100): 
            print RandomGraphGenerator.newman_watts_strogatz_graph, n, i
            graphsToReturn.append([i*TIME_UNIT_IN_SECONDS, my_nx.getDictForGraph(newman_watts_strogatz_graph(n,k,p))])
        return graphsToReturn
    @staticmethod
    def powerlawClusterGraph(n,m=3,p=0.3):
        graphsToReturn = []
        for i in range(100): 
            print RandomGraphGenerator.powerlaw_cluster_graph, n, i
            graphsToReturn.append([i*TIME_UNIT_IN_SECONDS, my_nx.getDictForGraph(powerlaw_cluster_graph(n,m,p))])
        return graphsToReturn
    @staticmethod
    def getGraphs(n, graphType):
        for data in FileIO.iterateJsonFromFile(randomGraphsFolder%graphType):
            if n==data['n']: 
                graphs = []
                for k,g in data['graphs']:
                    graph = my_nx.getGraphFromDict(g)
                    for n in graph.nodes()[:]: graph.node[n]['w']=1
                    for u,v in graph.edges()[:]: graph.edge[u][v]['w']=1
                    graphs.append((k, graph))
                return graphs
    @staticmethod
    def run():
        for graphType, method in [\
#                                  (RandomGraphGenerator.fast_gnp_random_graph, RandomGraphGenerator.fastGnp),
#                                  (RandomGraphGenerator.erdos_renyi_graph, RandomGraphGenerator.erdosRenyi),
#                                  (RandomGraphGenerator.newman_watts_strogatz_graph, RandomGraphGenerator.nWS),
                                (RandomGraphGenerator.powerlaw_cluster_graph, RandomGraphGenerator.powerlawClusterGraph),
                                  ]:
            for i in range(1, 11): FileIO.writeToFileAsJson({'n': 100*i, 'graphs': method(1000*i)}, randomGraphsFolder%graphType)

def getGraphs(area, timeRange): return sorted([(d['ep'], my_nx.getGraphFromDict(d['graph']))for d in FileIO.iterateJsonFromFile(epochGraphsFile%(area, '%s_%s'%timeRange))])
def tempGetGraphs(area, timeRange): return sorted([(d['ep'], my_nx.getGraphFromDict(d['graph']))for d in FileIO.iterateJsonFromFile(tempEpochGraphsFile%(area, '%s_%s'%timeRange))])
def writeTempGraphs(area, timeRange):
    dataToWrite = sorted([(d['ep'], d)for d in FileIO.iterateJsonFromFile(epochGraphsFile%(area, '%s_%s'%timeRange))])[:100]
    for ep, d in dataToWrite: FileIO.writeToFileAsJson(d, tempEpochGraphsFile%(area, '%s_%s'%timeRange))
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
#        for i in graphIdsToCombine:
#            ep, l = i.split('_')
#            print i, datetime.datetime.fromtimestamp(float(ep)), l, graphMap[i].number_of_nodes()
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
            graphMap['%s_%s'%(graphId, 1)] = graphMap[graphId]
        for k in graphMap.keys()[:]:
            if '_' not in str(k): del graphMap[k]
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
                print graphType, linear, graph.number_of_nodes(), j, runningTime
                dataToReturn.append([intervalInSeconds, runningTime, graph.number_of_nodes()])
            return dataToReturn
        graphFile = runningTimesFolder%graphType
        print graphFile
        GeneralMethods.runCommand('rm -rf %s'%graphFile)
        for linear in [False, True]: FileIO.writeToFileAsJson({'linear': linear, 'running_time': getRunningTime(graphs, linear)}, graphFile)
    @staticmethod
    def plotRunningTime(graphType):
        for data in FileIO.iterateJsonFromFile(runningTimesFolder%graphType):
            dataX, dataY, _ = zip(*data['running_time'])
            dataX = map(lambda x: x/(24*60*60), dataX)
            label, marker = 'linear', 'o'
            if not data['linear']: label, marker = 'logarithmic', 'x'
#            dataX, dataY = splineSmooth(dataX, dataY)
            plt.plot(dataX, dataY, marker=marker, label=label, lw=2)
        plt.legend(loc=2)
        plt.title('Running time comparison')
        plt.xlabel('Interval width (days)')
        plt.ylabel('Running Time (s)')
        plt.show()
    @staticmethod
    def run():
        timeRange, dataType, area = (5,11), 'world', 'world'
        LocationGraphs.runningTimeAnalysis(getGraphs(area, timeRange), 'location')
#        LocationGraphs.runningTimeAnalysis(RandomGraphGenerator.getGraphs(100, RandomGraphGenerator.erdos_renyi_graph), RandomGraphGenerator.erdos_renyi_graph)
        LocationGraphs.plotRunningTime('location')
#        LocationGraphs.plotRunningTime(RandomGraphGenerator.powerlaw_cluster_graph)

def temp_analysis():
    graphMap = dict(tempGetGraphs(area, timeRange))
    startingGraphId, endingGraphId = min(graphMap.keys()), max(graphMap.keys())
    LocationGraphs.updateLogarithmicGraphs(graphMap)
    linearGraph = LocationGraphs.combineLocationGraphs(graphMap, startingGraphId, datetime.datetime.fromtimestamp(endingGraphId+1), TIME_UNIT_IN_SECONDS*10, linear=True)
    logarithmicGraph = LocationGraphs.combineLocationGraphs(graphMap, startingGraphId, datetime.datetime.fromtimestamp(endingGraphId+1), TIME_UNIT_IN_SECONDS*10, linear=False)
    print linearGraph.number_of_nodes()
    print logarithmicGraph.number_of_nodes()

def getInputFiles(months, folderType='/'): return [hdfsInputFolder+folderType+'/'+str(m) for m in months] 
def mr_task(timeRange, dataType, area):
#    runMRJob(MRGraph, hashtagsFile%(area, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), dataType), jobconf={'mapred.reduce.tasks':160})
    runMRJob(MRGraph, epochGraphsFile%(area, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), dataType), jobconf={'mapred.reduce.tasks':160})   

if __name__ == '__main__':
#    timeRange, dataType, area = (5,6), 'world', 'us'
    timeRange, dataType, area = (5,11), 'world', 'world'
    
#    mr_task(timeRange, dataType, area)
#    temp_analysis()
    LocationGraphs.run()
#    RandomGraphGenerator.run()