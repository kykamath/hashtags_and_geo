'''
Created on Dec 15, 2011

@author: kykamath
'''
from library.mrjobwrapper import ModifiedMRJob
import cjson, time
from library.geo import getLattice, getLatticeLid
from library.twitter import getDateTimeObjectFromTweetTimestamp
from library.classes import GeneralMethods
from collections import defaultdict
from itertools import groupby, combinations
from operator import itemgetter
import networkx as nx
from library.graphs import Networkx as my_nx

LATTICE_ACCURACY = 0.145
TIME_UNIT_IN_SECONDS = 60*60

MIN_HASHTAG_OCCURENCES = 10
MIN_OCCURANCES_TO_ASSIGN_HASHTAG_TO_A_LOCATION = 10

def iterateHashtagObjectInstances(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t =  GeneralMethods.approximateEpoch(time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple()), TIME_UNIT_IN_SECONDS)
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
        def updateNode(graph, u, w):
            if graph.has_node(u): graph.node[u]['w']+=w
            else: graph.add_node(u, {'w': w})
        def updateEdge(graph, u, v, w):
            if graph.has_edge(u,v): graph.edge[u][v]['w']+=w
            else: graph.add_edge(u, v, {'w': w})
        graph = nx.Graph()
        for ep, instances in self.epochs.iteritems(): 
            occurances = dict([(h, map(itemgetter(0), occs)) for h, occs in groupby(instances, key=itemgetter(1))])
            for h in occurances.keys()[:]: 
                hashtagsMap = dict(filter(lambda t: t[1]>=MIN_OCCURANCES_TO_ASSIGN_HASHTAG_TO_A_LOCATION, [(lid, len(list(l)))for lid, l in groupby(occurances[h])]))
                if hashtagsMap and len(hashtagsMap)>1: 
                    for u, v in combinations(hashtagsMap,2):  updateNode(graph, u, hashtagsMap[u]), updateNode(graph, v, hashtagsMap[v]), updateEdge(graph, u, v, min([hashtagsMap[u], hashtagsMap[v]]))
            if graph.edges(): 
                totalEdgeWeight = sum([d['w'] for _,_,d in graph.edges(data=True)])+0.0
                for u,v in graph.edges()[:]: graph[u][v]['w']/=totalEdgeWeight
                yield ep, {'ep': ep, 'graph': my_nx.getDictForGraph(graph)}
    def groupOccurrencesByEpochReduce(self, key, epochObject):
        yield key, list(epochObject)
    # Tasks
    def jobsToGetHastagObjectsWithEndingWindow(self): return [self.mr(mapper=self.parse_hashtag_objects, mapper_final=self.parse_hashtag_objects_final, reducer=self.combine_hashtag_instances_without_ending_window)]
    def jobsToGetEpochGraph(self): return self.jobsToGetHastagObjectsWithEndingWindow() + \
                                                [self.mr(mapper=self.groupOccurrencesByEpochMap, mapper_final=self.groupOccurrencesByEpochMapFinal, reducer=self.groupOccurrencesByEpochReduce)]
    def steps(self):
#        return self.jobsToGetHastagObjectsWithEndingWindow()
        return self.jobsToGetEpochGraph()
if __name__ == '__main__':
    MRGraph.run()