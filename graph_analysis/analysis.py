'''
Created on Dec 15, 2011

@author: kykamath
'''
import sys, datetime
from library.geo import getLocationFromLid, plotPointsOnWorldMap
sys.path.append('../')
from library.mrjobwrapper import runMRJob
from library.file_io import FileIO
from graph_analysis.mr_modules import MRGraph
from graph_analysis.settings import hdfsInputFolder, epochGraphsFile,\
    hashtagsFile
import networkx as nx
from library.graphs import Networkx as my_nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plotLocationGraphOnMap(graph):
    for node in graph.nodes():
        points = map(lambda  lid:getLocationFromLid(lid.replace('_', ' ')), graph.nodes())
        _, m =plotPointsOnWorldMap(points, s=10, lw=0, c='m', returnBaseMapObject=True)
        totalEdgeWeight = max([d['w'] for _,_,d in graph.edges(node, data=True)])+0.0
        for u, v, data in graph.edges(node, data=True):
            u, v, w = getLocationFromLid(u.replace('_', ' ')), getLocationFromLid(v.replace('_', ' ')), data['w']
            m.drawgreatcircle(u[1],u[0],v[1],v[0],color=cm.Purples(w/totalEdgeWeight), alpha=0.5)
        
        plt.show()
    exit()
def getGraphs(timeRange): return sorted([(d['ep'], my_nx.getGraphFromDict(d['graph']))for d in FileIO.iterateJsonFromFile(epochGraphsFile%('%s_%s'%timeRange))])

def temp_analysis():
    for i, (ep, graph) in enumerate(getGraphs(timeRange)):
        print i, plotLocationGraphOnMap(graph)
#    for e in sorted(epochs):
#        print datetime.datetime.fromtimestamp(e)

def getInputFiles(months, folderType='/'): return [hdfsInputFolder+folderType+'/'+str(m) for m in months] 
def mr_task(timeRange, dataType, area):
#    runMRJob(MRGraph, hashtagsFile%(area, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), dataType), jobconf={'mapred.reduce.tasks':160})
    runMRJob(MRGraph, epochGraphsFile%(area, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), dataType), jobconf={'mapred.reduce.tasks':160})   

if __name__ == '__main__':
    timeRange = (2,3)
    dataType, area = 'world', 'us'
    mr_task(timeRange, dataType, area)
#    temp_analysis()