'''
Created on Dec 15, 2011

@author: kykamath
'''
import sys, datetime
import math
from networkx.generators.random_graphs import erdos_renyi_graph,\
    fast_gnp_random_graph, newman_watts_strogatz_graph, powerlaw_cluster_graph
import time
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
    hashtagsFile, us_boundary, runningTimesFolder, randomGraphsFolder,\
    tempEpochGraphsFile, qualityMetricsFolder
import networkx as nx
import networkx.algorithms.isomorphism as iso
from operator import itemgetter
from library.graphs import Networkx as my_nx
from library.graphs import clusterUsingAffinityPropagation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import defaultdict
from library.plotting import splineSmooth, smooth

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
    _, m =plotPointsOnUSMap(points, s=0, lw=0, c=colors, returnBaseMapObject=True)
    for u, v, data in graph.edges(data=True):
        if nodeToClusterIdMap[u]==nodeToClusterIdMap[v]:
            color, u, v, w = colorMap[nodeToClusterIdMap[u]], getLocationFromLid(u.replace('_', ' ')), getLocationFromLid(v.replace('_', ' ')), data['w']
            m.drawgreatcircle(u[1],u[0],v[1],v[0],color=color, alpha=0.5)
    plt.show()

def combineGraphList(graphs, edgesToKeep=1.0):
    def createSortedGraph(g):
        gToReturn = nx.Graph()
        for u in sorted(g.nodes()): gToReturn.add_node(u, {'w': g.node[u]['w']})
        for u, v in g.edges(): gToReturn.add_edge(u, v, {'w': g.edge[u][v]['w']})
        assert iso.is_isomorphic(gToReturn,g, edge_match=lambda e1,e2: e1['w']==e2['w'], node_match=lambda u,v: u['w']==v['w'])
        return gToReturn
    graph = nx.Graph()
    def addToG(g):
        nodesUpdated = set()
        for u,v,data in g.edges(data=True): 
            if u not in nodesUpdated: updateNode(graph, u, g.node[u]['w']), nodesUpdated.add(u)
            if v not in nodesUpdated: updateNode(graph, v, g.node[v]['w']), nodesUpdated.add(v)
            updateEdge(graph, u, v, data['w'])
    for g in graphs: addToG(g)
    edgesToRemove = sorted([(u,v, data['w']) for u,v,data in graph.edges(data=True)], key=itemgetter(2))[:int(graph.number_of_edges()*(1-edgesToKeep))]
    for u,v,_ in edgesToRemove: graph.remove_edge(u,v)
    graph = createSortedGraph(graph)
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
#        print graphIdsToCombine
#        for i in graphIdsToCombine:
#            ep, l = i.split('_')
#            print i, datetime.datetime.fromtimestamp(float(ep)), l, graphMap[i].number_of_nodes()
        graphsToCombine = [graphMap[id] for id in graphIdsToCombine]
        return combineGraphList(graphsToCombine, **kwargs)
    @staticmethod
    def updateLogarithmicGraphs(graphMap, **kwargs):
        print 'Building logarithmic graphs... ',
        startingGraphId = sorted(graphMap.keys())[0]
        for graphId in sorted(graphMap.keys()):
            i = LocationGraphs.getLogarithmicGraphId(startingGraphId, graphId)
            if i%2==0: 
                indices = map(lambda j: 2**j, filter(lambda j: i%(2**j)==0, range(1, int(math.log(i+1,2))+1)))
                for graphIdsToCombine in [map(lambda j: graphId-j*TIME_UNIT_IN_SECONDS, range(index)) for index in indices]:
                    graphsToCombine = [graphMap[j] for j in graphIdsToCombine if j in graphMap]
                    graphMap['%s_%s'%(graphId, len(graphIdsToCombine))] = combineGraphList(graphsToCombine, **kwargs)
            graphMap['%s_%s'%(graphId, 1)] = graphMap[graphId]
        for k in graphMap.keys()[:]:
            if '_' not in str(k): del graphMap[k]
        print 'Completed!!'
        return startingGraphId
    @staticmethod
    def analyzeRunningTime(graphs, graphType, numberOfPoints=50):
        edgesToKeep = 0.35
        def getRunningTime(graphs, linear):
            graphMap = dict(graphs)
            startingGraphId, endingGraphId = min(graphMap.keys()), max(graphMap.keys())
            timeDifference = endingGraphId-startingGraphId
            LocationGraphs.updateLogarithmicGraphs(graphMap, edgesToKeep=edgesToKeep)
            dataToReturn = []
            for j, intervalInSeconds in enumerate(range(0, timeDifference, int(timeDifference/numberOfPoints))):
                ts = time.time()
                graph = LocationGraphs.combineLocationGraphs(graphMap, startingGraphId, datetime.datetime.fromtimestamp(endingGraphId+1), intervalInSeconds, linear=linear,  edgesToKeep=edgesToKeep)
                noOfClusters, clusters = clusterUsingAffinityPropagation(graph)
                clusters = [[str(c), [l[0]for l in lst]] for c, lst in groupby(sorted(clusters, key=itemgetter(1)), key=itemgetter(1))]
                te = time.time()
                edgeWeights = sum(data['w'] for _,_,data in graph.edges(data=True))
                print graphType, linear, len(clusters), graph.number_of_nodes(), graph.number_of_edges(), edgeWeights, j, te-ts
                dataToReturn.append({'intervalInSeconds': intervalInSeconds, 'runningTime': te-ts, 'clusters': clusters, 'noOfNodes': graph.number_of_nodes()})
            return dataToReturn
        graphFile = runningTimesFolder%graphType
        print graphFile
        GeneralMethods.runCommand('rm -rf %s'%graphFile)
        for linear in [False, True]: FileIO.writeToFileAsJson({'linear': linear, 'analysis': getRunningTime(graphs, linear)}, graphFile)
    @staticmethod
    def analyzeQuality(graphs, graphType):
        def getQualityScore(graphMap, edgesToKeep, timeDifference):
            dataToReturn = []
            for j, intervalInSeconds in enumerate([1]):
                intervalInSeconds*=timeDifference
                linearGraph = LocationGraphs.combineLocationGraphs(graphMap, startingGraphId, datetime.datetime.fromtimestamp(endingGraphId+1), intervalInSeconds, linear=True, edgesToKeep=edgesToKeep)
                logGraph = LocationGraphs.combineLocationGraphs(graphMap, startingGraphId, datetime.datetime.fromtimestamp(endingGraphId+1), intervalInSeconds, linear=False, edgesToKeep=edgesToKeep)
                linearClusters = [[str(c), [l[0]for l in lst]] for c, lst in groupby(sorted(clusterUsingAffinityPropagation(linearGraph)[1], key=itemgetter(1)), key=itemgetter(1))]
                logarithmicClusters = [[str(c), [l[0]for l in lst]] for c, lst in groupby(sorted(clusterUsingAffinityPropagation(logGraph)[1], key=itemgetter(1)), key=itemgetter(1))]
                score = LocationGraphs.getClusterQualityScore(linearClusters, logarithmicClusters)
                print intervalInSeconds, edgesToKeep, score
                dataToReturn.append(score)
            return dataToReturn
        graphFile = qualityMetricsFolder%graphType
        print graphFile
        GeneralMethods.runCommand('rm -rf %s'%graphFile)
        for edgesToKeep in range(1,11): 
#        for edgesToKeep in [1,10]: 
            edgesToKeep*=0.1
            graphMap = dict(graphs[:])
            startingGraphId, endingGraphId = min(graphMap.keys()), max(graphMap.keys())
            timeDifference = endingGraphId-startingGraphId
            LocationGraphs.updateLogarithmicGraphs(graphMap, edgesToKeep=edgesToKeep)
#            print {'edgesToKeep': edgesToKeep, 'score': np.mean(getQualityScore(graphMap, edgesToKeep, timeDifference))}
            FileIO.writeToFileAsJson({'edgesToKeep': edgesToKeep, 'score': np.mean(getQualityScore(graphMap, edgesToKeep, timeDifference))}, graphFile)
    @staticmethod
    def plotRunningTime(graphType):
        for data in FileIO.iterateJsonFromFile(runningTimesFolder%graphType):
            dataX, dataY = zip(*[(d['intervalInSeconds'], d['runningTime']) for d in data['analysis']])
            dataX = map(lambda x: x/(24*60*60), dataX)
            label, marker = 'linear', 'o'
            if not data['linear']: label, marker = 'logarithmic', 'x'
#            dataX, dataY = splineSmooth(dataX, dataY)
#            smooth(dataY, 4)
            dataY = smooth(dataY, 15)[:len(dataX)]
#            dataX, dataY = dataX[5:], dataY[5:]
            plt.plot(dataX, dataY, marker=marker, label=label, lw=2)
            plt.xlim(xmin=3)
        plt.legend(loc=2)
        plt.title(graphType)
        plt.xlabel('Interval width (days)')
        plt.ylabel('Running Time (s)')
#        plt.show()
        plt.savefig('images/rt_%s.png'%graphType)
    @staticmethod
    def plotHotspotsQuality(graphType):
        dataX, dataY = zip(* [data.values() for data in FileIO.iterateJsonFromFile(qualityMetricsFolder%graphType)])
#        dataY = smooth(dataY, 3)[:len(dataX)]
        plt.plot(dataX, dataY, c='k', lw=1)
        plt.plot(dataX, dataY, 'o', c='r', markersize=15)
        plt.ylim(ymin=0.0, ymax=1.1)
        plt.title(graphType)
        plt.xlabel('Percentage of edges')
        plt.ylabel('Rand Score')
#        plt.show()
        plt.savefig('images/quality_%s.png'%graphType)
    @staticmethod
    def getClusterQualityScore(linearClusters, logarithmicClusters):
        linearClusters = [(id, set(cl)) for id, cl in linearClusters]
        logarithmicClusters = [(id, set(cl)) for id, cl in logarithmicClusters]
        nodeToClusterIdMap = dict([(n, [id]) for id, cl in linearClusters for n in cl])
        logToLinearClusterMap = {}
        for logarithmicClusterId, logarithmicCluster in logarithmicClusters:
            linearClusterId, linearCluster = max(linearClusters, key=lambda t: len(t[1].intersection(logarithmicCluster)))
            score = len(linearCluster.intersection(logarithmicCluster))/float(min(len(linearCluster), len(logarithmicCluster)))
            if score>=0.5:
                logToLinearClusterMap[logarithmicClusterId]=(linearClusterId,  len(linearCluster.intersection(logarithmicCluster)), len(linearCluster), len(logarithmicCluster))
                for n in logarithmicCluster: nodeToClusterIdMap[n].append(linearClusterId)
        nodeToClusterIdMap = dict(filter(lambda l: len(l[1])>1, nodeToClusterIdMap.iteritems()))
        labels_true, labels_pred = zip(*nodeToClusterIdMap.values())
        from sklearn import metrics
        return metrics.adjusted_rand_score(labels_true, labels_pred)
    @staticmethod
    def run():
        timeRange, dataType, area = (5,7), 'world', 'world'
        type, graphs = 'location_%s_%s'%timeRange, None#getGraphs(area, timeRange)
#        type, graphs = RandomGraphGenerator.powerlaw_cluster_graph, None# RandomGraphGenerator.getGraphs(100, RandomGraphGenerator.erdos_renyi_graph)
#        LocationGraphs.analyzeRunningTime(graphs, type, numberOfPoints=50)
#        LocationGraphs.analyzeQuality(graphs, type)
#        LocationGraphs.plotRunningTime(type)
        LocationGraphs.plotHotspotsQuality(type)

def temp_analysis():
    pass

def getInputFiles(months, folderType='/'): return [hdfsInputFolder+folderType+'/'+str(m) for m in months] 
def mr_task(timeRange, dataType, area):
#    runMRJob(MRGraph, hashtagsFile%(area, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), dataType), jobconf={'mapred.reduce.tasks':160})
    runMRJob(MRGraph, epochGraphsFile%(area, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), dataType), jobconf={'mapred.reduce.tasks':160})   

if __name__ == '__main__':
    timeRange, dataType, area = (2,11), 'world', 'us'
#    timeRange, dataType, area = (5,8), 'world', 'world'
#    timeRange, dataType, area = (5,11), 'world', 'world'
    
    mr_task(timeRange, dataType, area)
#    temp_analysis()
#    LocationGraphs.run()
#    RandomGraphGenerator.run()