'''
Created on Dec 15, 2011

@author: kykamath
'''
import sys, datetime
from library.geo import getLocationFromLid, plotPointsOnWorldMap,\
    plotPointsOnUSMap, isWithinBoundingBox
from library.classes import GeneralMethods
sys.path.append('../')
from library.mrjobwrapper import runMRJob
from itertools import groupby
from library.file_io import FileIO
from graph_analysis.mr_modules import MRGraph
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

def getGraphs(area, timeRange): return sorted([(d['ep'], my_nx.getGraphFromDict(d['graph']))for d in FileIO.iterateJsonFromFile(epochGraphsFile%(area, '%s_%s'%timeRange))])
def temp_analysis():
    for i, (ep, graph) in enumerate(getGraphs(area, timeRange)):
        print datetime.datetime.fromtimestamp(ep)
        plotLocationClustersOnMap(graph)
#        clusters = clusterUsingAffinityPropagation(graph)[1]
#        clusters = [(c, GeneralMethods.getRandomColor(), list(l)) for c, l in groupby(sorted(clusters, key=itemgetter(1)), key=itemgetter(1))]
#        print datetime.datetime.fromtimestamp(ep) , graph.number_of_nodes(), sorted(clusters, key=itemgetter(1), reverse=True)
#        exit()
#        plotLocationGraphOnMap(graph)
#    for e in sorted(epochs):
#        print datetime.datetime.fromtimestamp(e)

def getInputFiles(months, folderType='/'): return [hdfsInputFolder+folderType+'/'+str(m) for m in months] 
def mr_task(timeRange, dataType, area):
#    runMRJob(MRGraph, hashtagsFile%(area, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), dataType), jobconf={'mapred.reduce.tasks':160})
    runMRJob(MRGraph, epochGraphsFile%(area, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), dataType), jobconf={'mapred.reduce.tasks':160})   

if __name__ == '__main__':
#    timeRange, dataType, area = (5,6), 'world', 'us'
    timeRange, dataType, area = (5,6), 'world', 'world'
    
    mr_task(timeRange, dataType, area)
#    temp_analysis()