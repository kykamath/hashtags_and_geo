'''
Created on Dec 8, 2011

@author: kykamath
'''
from library.file_io import FileIO
from settings import hashtagsWithoutEndingWindowFile, hashtagsLatticeGraphFile
from experiments.mr_area_analysis import getOccuranesInHighestActiveRegion,\
    TIME_UNIT_IN_SECONDS, LATTICE_ACCURACY
import numpy as np
from library.stats import getOutliersRangeUsingIRQ
from library.geo import getHaversineDistanceForLids, getLatticeLid
from collections import defaultdict
from operator import itemgetter
import networkx as nx
from library.graphs import plot

def filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeHashtags, neighborHashtags, findLag=True):
    if findLag: 
        dataToReturn = [(hashtag, np.abs(latticeHashtags[hashtag][0]-timeTuple[0])/TIME_UNIT_IN_SECONDS) for hashtag, timeTuple in neighborHashtags.iteritems() if hashtag in latticeHashtags]
        _, upperRangeForTemporalDistance = getOutliersRangeUsingIRQ(zip(*(dataToReturn))[1])
        return dict(filter(lambda t: t[1]<=upperRangeForTemporalDistance, dataToReturn))
    else: 
        dataToReturn = [(hashtag, timeTuple, np.abs(latticeHashtags[hashtag][0]-timeTuple[0])/TIME_UNIT_IN_SECONDS) for hashtag, timeTuple in neighborHashtags.iteritems() if hashtag in latticeHashtags]
        _, upperRangeForTemporalDistance = getOutliersRangeUsingIRQ(zip(*(dataToReturn))[2])
        return dict([(t[0], t[1]) for t in dataToReturn if t[2]<=upperRangeForTemporalDistance])
def latticeNodeByHaversineDistance(latticeObject):
    dataToReturn = {'id': latticeObject['id'], 'links': {}}
    for neighborLattice, _ in latticeObject['links'].iteritems(): dataToReturn['links'][neighborLattice]=getHaversineDistanceForLids(latticeObject['id'].replace('_', ' '), neighborLattice.replace('_', ' '))
    return dataToReturn
def latticeNodeBySharingProbability(latticeObject):
    dataToReturn = {'id': latticeObject['id'], 'links': {}}
    latticeHashtagsSet = set(latticeObject['hashtags'])
    for neighborLattice, neighborHashtags in latticeObject['links'].iteritems():
        neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags)
        neighborHashtagsSet = set(neighborHashtags)
        dataToReturn['links'][neighborLattice]=len(latticeHashtagsSet.intersection(neighborHashtagsSet))/float(len(latticeHashtagsSet))
    return dataToReturn
def latticeNodeByTemporalDistanceInHours(latticeObject):
    dataToReturn = {'id': latticeObject['id'], 'links': {}}
    for neighborLattice, neighborHashtags in latticeObject['links'].iteritems():
        neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags)
        dataX = zip(*neighborHashtags.iteritems())[1]
        dataToReturn['links'][neighborLattice]=np.mean(dataX)
    return dataToReturn
def latticeNodeByTemporalClosenessScore(latticeObject):
    dataToReturn = {'id': latticeObject['id'], 'links': {}}
    for neighborLattice, neighborHashtags in latticeObject['links'].iteritems():
        neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags)
        dataX = map(lambda lag: LatticeGraph.temporalScore(lag, LatticeGraph.upperRangeForTemporalDistances), zip(*neighborHashtags.iteritems())[1])
        dataToReturn['links'][neighborLattice]=np.mean(dataX)
    return dataToReturn
def latticeNodeByHashtagDiffusionLocationVisitation(latticeObject, generateTemporalClosenessScore=False):
    def updateHashtagDistribution(latticeObj, hashtagDistribution): 
        for h, (t, _) in latticeObj['hashtags'].iteritems(): hashtagDistribution[h].append([latticeObj['id'], t])
    dataToReturn = {'id': latticeObject['id'], 'links': {}}
    hashtagDistribution, numberOfHastagsAtCurrentLattice = defaultdict(list), float(len(latticeObject['hashtags']))
    updateHashtagDistribution(latticeObject, hashtagDistribution)
    for neighborLattice, neighborHashtags in latticeObject['links'].iteritems(): 
        neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags, findLag=False)
        updateHashtagDistribution({'id':neighborLattice, 'hashtags': neighborHashtags}, hashtagDistribution)
    latticeEdges = {'in': defaultdict(float), 'out': defaultdict(float)}
    for hashtag in hashtagDistribution.keys()[:]: 
        hashtagOccurences = sorted(hashtagDistribution[hashtag], key=itemgetter(1))
        for neighborLattice, lag in [(t[0], (t[1]-latticeObject['hashtags'][hashtag][0])/TIME_UNIT_IN_SECONDS) for t in hashtagOccurences if t[0]!=latticeObject['id']]:
            if not generateTemporalClosenessScore:
                if lag<0: latticeEdges['in'][neighborLattice]+=-lag/numberOfHastagsAtCurrentLattice
                else: latticeEdges['out'][neighborLattice]+=lag/numberOfHastagsAtCurrentLattice
            else: 
                if lag<0: latticeEdges['in'][neighborLattice]+=LatticeGraph.temporalScore(-lag, LatticeGraph.upperRangeForTemporalDistances)/numberOfHastagsAtCurrentLattice
                else: latticeEdges['out'][neighborLattice]+=LatticeGraph.temporalScore(lag, LatticeGraph.upperRangeForTemporalDistances)/numberOfHastagsAtCurrentLattice
    dataToReturn['links']=latticeEdges
    return dataToReturn
class LatticeGraph:
    upperRangeForTemporalDistances = 8.24972222222
    typeSharingProbability = {'id': 'sharing_probability', 'method': latticeNodeBySharingProbability, 'title': 'Probability of sharing hastags'}
    typeTemporalCloseness = {'id': 'temporal_closeness', 'method': latticeNodeByTemporalClosenessScore, 'title': 'Temporal closeness'}
    typeTemporalDistanceInHours = {'id': 'temporal_distance_in_hours', 'method': latticeNodeByTemporalDistanceInHours, 'title': 'Temporal distance (hours)'}
    typeHaversineDistance = {'id': 'haversine_distance', 'method': latticeNodeByHaversineDistance, 'title': 'Distance (miles)'}
    def __init__(self, graphFile, latticeGraphType):
        self.graphFile = graphFile
        self.latticeGraphType = latticeGraphType
    def load(self):
        i = 1
        self.graph = nx.DiGraph()
        for latticeObject in FileIO.iterateJsonFromFile(self.graphFile):
            print i; i+=1
            latticeObject = self.latticeGraphType['method'](latticeObject)
            if 'in' in latticeObject['links']:
                for no, w in latticeObject['links']['in'].iteritems(): self.graph.add_edge(no, latticeObject['id'], {'w': w})
                for no, w in latticeObject['links']['out'].iteritems(): self.graph.add_edge(latticeObject['id'], no, {'w': w})
            else:
                for no, w in latticeObject['links'].iteritems(): self.graph.add_edge(latticeObject['id'], no, {'w': w}), self.graph.add_edge(no, latticeObject['id'], {'w': w})
            if i==100: break
#        plot(self.graph)
        return self.graph
    @staticmethod
    def normalizeNode(latticeNodeObject):
        if 'in' in latticeNodeObject['links']:
            for edgeType, edges in latticeNodeObject['links'].iteritems():
                totalEdgeWeight = sum(edges.values())
                for lattice, score in edges.items()[:]: latticeNodeObject['links'][edgeType][lattice] = score/totalEdgeWeight
        else:
            totalEdgeWeight = sum(latticeNodeObject['links'].values())
            for lattice, score in latticeNodeObject['links'].items()[:]: latticeNodeObject['links'][lattice]=score/totalEdgeWeight
    @staticmethod
    def temporalScore(lag, width):
        lag=int(lag*TIME_UNIT_IN_SECONDS)
        width=int(width*TIME_UNIT_IN_SECONDS)
        if lag==0: return 1.0
        elif lag>=width: return 0.0
        return 1-np.log(lag)/np.log(width)
    @staticmethod
    def run(timeRange, folderType):
        latticeGraph = LatticeGraph(hashtagsLatticeGraphFile%(folderType,'%s_%s'%timeRange), LatticeGraph.typeTemporalCloseness)
        latticeGraph.load()
        print latticeGraph.graph.number_of_nodes()
        
class Customer:
    pass

class Hashtag:
    def __init__(self, hashtagObject): 
        self.hashtagObject = hashtagObject
        self.occuranceDistributionInLattices = defaultdict(list)
        self.latestObservedOccuranceTime, self.latestObservedWindow = None, None
        self.nextOccurenceIterator = self.getNextOccurance()
    def getNextOccurance(self):
        for oc in getOccuranesInHighestActiveRegion(self.hashtagObject): 
            latestObservedOccurance = [getLatticeLid(oc[0], accuracy=LATTICE_ACCURACY), oc[1]]
            self.latestObservedOccuranceTime = latestObservedOccurance[1]
            yield latestObservedOccurance
#            yield self.currentOccurance
    def getOccrancesForNextTimeWindow(self, timeWindowInSeconds):
        occurancesToReturn = []
        if not self.latestObservedWindow: 
            occurancesToReturn.append(self.nextOccurenceIterator.next())
            self.latestObservedWindow = occurancesToReturn[0][1]
#        occurancesToReturn = [self.currentOccurance]
        self.latestObservedWindow+=timeWindowInSeconds
        while self.latestObservedOccuranceTime<=self.latestObservedWindow: occurancesToReturn.append(self.nextOccurenceIterator.next())
        return occurancesToReturn
    def updateOccuranceDistributionInLattices(self, occurrances): [self.occuranceDistributionInLattices[oc[0]].append(oc[1]) for oc in occurrances]
    @staticmethod
    def iterateHashtags(timeRange, folderType):
        for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(folderType,'%s_%s'%timeRange)): yield Hashtag(h)

class Simulation:
    TIME_WINDOW_IN_SECONDS = 5*60
    @staticmethod
    def runModel(customerModel, latticeGraph, hashtagsIterator):
        currentLattices = latticeGraph.nodes()
        for hashtag in hashtagsIterator:
            print [h for h in hashtag.getOccrancesForNextTimeWindow(Simulation.TIME_WINDOW_IN_SECONDS) if h[0] in currentLattices]
#                print h
            exit()
#            print len(list(hashtag.getNextOccurance()))
    @staticmethod
    def run():
        timeRange, folderType = (2,11), 'world'
        graph = LatticeGraph(hashtagsLatticeGraphFile%(folderType,'%s_%s'%timeRange), LatticeGraph.typeTemporalCloseness).load()
        Simulation.runModel(Customer(), graph, Hashtag.iterateHashtags(timeRange, folderType))

if __name__ == '__main__':
    Simulation.run()