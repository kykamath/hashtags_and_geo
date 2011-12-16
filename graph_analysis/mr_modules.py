'''
Created on Dec 15, 2011

@author: kykamath
'''
from library.mrjobwrapper import ModifiedMRJob
import cjson, time
from library.geo import getLattice, getLatticeLid, isWithinBoundingBox
from library.twitter import getDateTimeObjectFromTweetTimestamp
from library.classes import GeneralMethods
from collections import defaultdict
from itertools import groupby, combinations
from operator import itemgetter
import networkx as nx
from library.graphs import Networkx as my_nx

#LATTICE_ACCURACY = 0.145
LATTICE_ACCURACY = 0.0725
TIME_UNIT_IN_SECONDS = 6*60*60

MIN_HASHTAG_OCCURENCES = 5
MIN_OCCURANCES_TO_ASSIGN_HASHTAG_TO_A_LOCATION = 3

#BOUNDARY = [[24.527135,-127.792969], [49.61071,-59.765625]] #US
BOUNDARY = [[-90,-180], [90, 180]] # World

def iterateHashtagObjectInstances(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t =  GeneralMethods.approximateEpoch(time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple()), TIME_UNIT_IN_SECONDS)
    if isWithinBoundingBox(l, BOUNDARY):
        point = getLatticeLid(l, LATTICE_ACCURACY)
        if point!='0.0000_0.0000':
            for h in data['h']: yield h.lower(), [point, t]
    
def getHashtagWithoutEndingWindow(key, values):
    occurences = []
    for instances in values: 
        for oc in instances['oc']: occurences.append(oc)
    if occurences:
#        e, l = min(occurences, key=lambda t: t[1]), max(occurences, key=lambda t: t[1])
        numberOfInstances=len(occurences)
        if numberOfInstances>=MIN_HASHTAG_OCCURENCES: 
            return {'h': key, 't': numberOfInstances, 'oc': sorted(occurences, key=lambda t: t[1])}

def updateNode(graph, u, w):
    if graph.has_node(u): graph.node[u]['w']+=w
    else: graph.add_node(u, {'w': w})
def updateEdge(graph, u, v, w):
    if graph.has_edge(u,v): graph.edge[u][v]['w']+=w
    else: graph.add_edge(u, v, {'w': w})

class MRGraph(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(MRGraph, self).__init__(*args, **kwargs)
        self.hashtags = defaultdict(list)
        self.epochs = defaultdict(list)
    def parse_hashtag_objects(self, key, line):
        if False: yield # I'm a generator!
        for h, d in iterateHashtagObjectInstances(line): self.hashtags[h].append(d)
    def parse_hashtag_objects_final(self):
        for h, instances in self.hashtags.iteritems(): # e = earliest, l = latest
            yield h, {'oc': instances, 'e': min(instances, key=lambda t: t[1]), 'l': max(instances, key=lambda t: t[1])}
    def combine_hashtag_instances_without_ending_window(self, key, values):
        hashtagObject = getHashtagWithoutEndingWindow(key, values)
        if hashtagObject: yield key, hashtagObject 
    ''' Group buy occurrence per epoch.
    '''
    def groupOccurrencesByEpochMap(self, key, hashtagObject):
        if False: yield # I'm a generator!
        for lid, ep in hashtagObject['oc']: self.epochs[ep].append([lid, hashtagObject['h']])
    def groupOccurrencesByEpochMapFinal(self):
        for ep, instances in self.epochs.iteritems(): 
            occurances = dict([(h, map(itemgetter(0), occs)) for h, occs in groupby(instances, key=itemgetter(1))])
            for h in occurances.keys()[:]: 
                hashtagsMap = [(lid, len(list(l)))for lid, l in groupby(occurances[h])]
                occurances[h] = hashtagsMap
            yield ep, occurances
    def groupOccurrencesByEpochReduceFinal(self, ep, epochObjects):
        graph, occurrences = nx.Graph(), defaultdict(list)
        for occDict in epochObjects:
            for h, occs in occDict.iteritems(): occurrences[h]+=occs
        for h in occurrences.keys()[:]: 
            hashtagsMap = dict(filter(lambda l: l[1]>=MIN_OCCURANCES_TO_ASSIGN_HASHTAG_TO_A_LOCATION, [(lid, sum(map(itemgetter(1), l)))for lid, l in groupby(occurrences[h], key=itemgetter(0))]))
            if hashtagsMap and len(hashtagsMap)>1: 
                nodesUpdated = set()
                for u, v in combinations(hashtagsMap,2):
                    if u not in nodesUpdated: updateNode(graph, u, hashtagsMap[u]), nodesUpdated.add(u)
                    if v not in nodesUpdated: updateNode(graph, v, hashtagsMap[v]), nodesUpdated.add(v)
                    updateEdge(graph, u, v, min([hashtagsMap[u], hashtagsMap[v]]))
        if graph.edges(): 
#            totalEdgeWeight = sum([d['w'] for _,_,d in graph.edges(data=True)])+0.0
#            for u,v in graph.edges()[:]: graph[u][v]['w']/=totalEdgeWeight
            yield ep, {'ep': ep, 'graph': my_nx.getDictForGraph(graph)} 
    # Tasks
    def jobsToGetHastagObjectsWithEndingWindow(self): return [self.mr(mapper=self.parse_hashtag_objects, mapper_final=self.parse_hashtag_objects_final, reducer=self.combine_hashtag_instances_without_ending_window)]
    def jobsToGetEpochGraph(self): return self.jobsToGetHastagObjectsWithEndingWindow() + \
                                                [self.mr(mapper=self.groupOccurrencesByEpochMap, mapper_final=self.groupOccurrencesByEpochMapFinal, reducer=self.groupOccurrencesByEpochReduceFinal)]
    def steps(self):
#        return self.jobsToGetHastagObjectsWithEndingWindow()
        return self.jobsToGetEpochGraph()
if __name__ == '__main__':
    MRGraph.run()