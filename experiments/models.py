'''
Created on Dec 8, 2011

@author: kykamath
'''
from library.file_io import FileIO
from settings import hashtagsWithoutEndingWindowFile, hashtagsLatticeGraphFile
from experiments.mr_area_analysis import getOccuranesInHighestActiveRegion,\
    TIME_UNIT_IN_SECONDS, LATTICE_ACCURACY, HashtagsClassifier,\
    getOccurranceDistributionInEpochs, CLASSIFIER_TIME_UNIT_IN_SECONDS,\
    getRadius
import numpy as np
from library.stats import getOutliersRangeUsingIRQ
from library.geo import getHaversineDistanceForLids, getLatticeLid
from collections import defaultdict
from operator import itemgetter
import networkx as nx
from library.graphs import plot
import datetime, math, random

GREEDY_LATTICE_SELECTION_MODEL = 'greedy'

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
    def __init__(self, graphFile, latticeGraphType, graphType=nx.Graph):
        self.graphFile = graphFile
        self.latticeGraphType = latticeGraphType
#        self.graphType = graphType
    def load(self):
        i = 1
        self.graph = nx.DiGraph()
        for latticeObject in FileIO.iterateJsonFromFile(self.graphFile):
            print i; i+=1
            latticeObject = self.latticeGraphType['method'](latticeObject)
            if 'in' in latticeObject['links']:
#                for no, w in latticeObject['links']['in'].iteritems(): self.graph.add_edge(no, latticeObject['id'], {'w': w})
#                for no, w in latticeObject['links']['out'].iteritems(): self.graph.add_edge(latticeObject['id'], no, {'w': w})
                no, w = max(latticeObject['links']['in'].iteritems(), key=itemgetter(1))
                self.graph.add_edge(no, latticeObject['id'], {'w': w})
                no, w = max(latticeObject['links']['out'].iteritems(), key=itemgetter(1))
                self.graph.add_edge(latticeObject['id'], no, {'w': w})
            else:
#                for no, w in latticeObject['links'].iteritems(): self.graph.add_edge(latticeObject['id'], no, {'w': w}), self.graph.add_edge(no, latticeObject['id'], {'w': w})
                no, w = max(latticeObject['links'].iteritems(), key=itemgetter(1))
                if not self.graph.has_edge(latticeObject['id'], no) or self.graph.edge[latticeObject['id']][no]['w']<w: 
                    self.graph.add_edge(latticeObject['id'], no, {'w': w}), self.graph.add_edge(no, latticeObject['id'], {'w': w})
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

class Metrics:
    @staticmethod
    def overallOccurancesHitRate(occuranceDistributionInLattices, occuranceDistributionInTargetLattices):
        totalOccurances, occurancesObserved = 0., 0.
        for k,v in occuranceDistributionInLattices.iteritems(): totalOccurances+=len(v)
        for k, v in occuranceDistributionInTargetLattices.iteritems(): occurancesObserved+=sum(v['occurances'].values())
        return occurancesObserved/totalOccurances
    @staticmethod
    def occurancesHitRateAfterTargetSelection(occuranceDistributionInLattices, occuranceDistributionInTargetLattices):
        totalOccurances, occurancesObserved = 0., 0.
        targetSelectionTimeUnit = min(v['selectedTimeUnit'] for v in occuranceDistributionInTargetLattices.values())
        for k,v in occuranceDistributionInLattices.iteritems(): totalOccurances+=len([i for i in v if i>targetSelectionTimeUnit])
        for k, v in occuranceDistributionInTargetLattices.iteritems(): occurancesObserved+=sum(v['occurances'].values())
        return occurancesObserved/totalOccurances
    @staticmethod
    def occurancesMissRateBeforeTargetSelection(occuranceDistributionInLattices, occuranceDistributionInTargetLattices):
        totalOccurances, occurancesBeforeTimeUnit = 0., 0.
        targetSelectionTimeUnit = min(v['selectedTimeUnit'] for v in occuranceDistributionInTargetLattices.values())
        for k,v in occuranceDistributionInLattices.iteritems(): totalOccurances+=len(v); occurancesBeforeTimeUnit+=len([i for i in v if i<=targetSelectionTimeUnit])
        return occurancesBeforeTimeUnit/totalOccurances
EvaluationMetrics = {
                     'overall_hit_rate': Metrics.overallOccurancesHitRate,
                     'hit_rate_after_target_selection': Metrics.occurancesHitRateAfterTargetSelection,
                     'miss_rate_before_target_selection': Metrics.occurancesMissRateBeforeTargetSelection
                     }

class LatticeSelectionModel:
    def __init__(self, id='random', budget=3, timeUnitToPickTargetLattices=6):
        self.budget = budget
        self.timeUnitToPickTargetLattices = timeUnitToPickTargetLattices
    def selectNextLatticesRandomly(self, currentTimeUnit, hashtag, graph, occuranceDistributionInLattices):
        if self.timeUnitToPickTargetLattices==currentTimeUnit: hashtag._initializeTargetLattices(currentTimeUnit, random.sample(occuranceDistributionInLattices, min([self.budget, len(occuranceDistributionInLattices)])))
class GreedyLatticeSelectionModel(LatticeSelectionModel):
    ''' Pick the location with maximum observations till that time.
    '''
    def __init__(self, budget=3, timeUnitToPickTargetLattices=6):
        super(GreedyLatticeSelectionModel, self).__init__(GREEDY_LATTICE_SELECTION_MODEL, budget, timeUnitToPickTargetLattices)
    def selectNextLatticesRandomly(self, currentTimeUnit, hashtag, graph, occuranceDistributionInLattices):
        if self.timeUnitToPickTargetLattices==currentTimeUnit: hashtag._initializeTargetLattices(currentTimeUnit, random.sample(occuranceDistributionInLattices, min([self.budget, len(occuranceDistributionInLattices)])))
                
def normalize(data):
    total = math.sqrt(float(sum([d**2 for d in data])))
    if total==0: return map(lambda d: 0, data)
    return map(lambda d: d/total, data)
class Hashtag:
    def __init__(self, hashtagObject, dataStructuresToBuildClassifier=False): 
        self.hashtagObject = hashtagObject
        self.occuranceDistributionInLattices = defaultdict(list)
        self.latestObservedOccuranceTime, self.latestObservedWindow = None, None
        self.nextOccurenceIterator = self._getNextOccurance()
        self.hashtagObject['oc'] = getOccuranesInHighestActiveRegion(self.hashtagObject)
        self.targetLattices = {}
        if self.hashtagObject['oc']: 
            self.timePeriod = (self.hashtagObject['oc'][-1][1]-self.hashtagObject['oc'][0][1])/TIME_UNIT_IN_SECONDS
            self.hashtagClassId = HashtagsClassifier.classify(self.hashtagObject)
        # Data structures for building classifier.
        if dataStructuresToBuildClassifier and self.isValidObject():
            self.classifiable = True
            occurranceDistributionInEpochs = getOccurranceDistributionInEpochs(getOccuranesInHighestActiveRegion(self.hashtagObject), timeUnit=CLASSIFIER_TIME_UNIT_IN_SECONDS, fillInGaps=True, occurancesCount=False)
            if occurranceDistributionInEpochs:
                self.occurances = zip(*sorted(occurranceDistributionInEpochs.iteritems(), key=itemgetter(0)))[1]
                self.occuranceCountVector = map(lambda t: len(t), self.occurances)
                self.occuranceLatticesVector = []
                for t in self.occurances:
                    if t: self.occuranceLatticesVector.append(getRadius(zip(*t)[0]))
                    else: self.occuranceLatticesVector.append(0.0)
            else: self.classifiable=False
    def getVector(self, length): 
        if len(self.occuranceCountVector)<length: 
            difference = length-len(self.occuranceCountVector)
            self.occuranceCountVector=self.occuranceCountVector+[0 for i in range(difference)]
            self.occuranceLatticesVector=self.occuranceLatticesVector+[0 for i in range(difference)]
        vector = normalize(self.occuranceCountVector[:length]) + self.occuranceLatticesVector[:length]
        return [vector, self.hashtagClassId]
    def isValidObject(self):
        if not self.hashtagObject['oc']: return False
        if not self.hashtagClassId: return False
        if self.timePeriod>HashtagsClassifier.stats[self.hashtagClassId]['outlierBoundary']: return False
        return True
    def _getNextOccurance(self):
        for oc in self.hashtagObject['oc']: 
            latestObservedOccurance = [getLatticeLid(oc[0], accuracy=LATTICE_ACCURACY), oc[1]]
            self.latestObservedOccuranceTime = latestObservedOccurance[1]
            yield latestObservedOccurance
    def getOccrancesEveryTimeWindowIterator(self, timeWindowInSeconds):
        while True:
            occurancesToReturn = []
            if not self.latestObservedWindow: 
                occurancesToReturn.append(self.nextOccurenceIterator.next())
                self.latestObservedWindow = occurancesToReturn[0][1]
            self.latestObservedWindow+=timeWindowInSeconds
            while self.latestObservedOccuranceTime<=self.latestObservedWindow: occurancesToReturn.append(self.nextOccurenceIterator.next())
            yield occurancesToReturn
    def updateOccuranceDistributionInLattices(self, currentTimeUnit, occurrances): [self.occuranceDistributionInLattices[oc[0]].append(currentTimeUnit) for oc in occurrances]
    def _initializeTargetLattices(self, currentTimeUnit, targetLattices):
        for lattice in targetLattices: 
            self.targetLattices[lattice] = {'selectedTimeUnit': None,'occurances':defaultdict(int)}
            self.targetLattices[lattice]['selectedTimeUnit'] = currentTimeUnit 
    def updateOccurancesInTargetLattices(self, currentTimeUnit, occuranceDistributionInLattices):
        for targetLattice in self.targetLattices:
            for occuranceTimeUnit in occuranceDistributionInLattices[targetLattice]: 
                if occuranceTimeUnit==currentTimeUnit: 
                    self.targetLattices[targetLattice]['occurances'][currentTimeUnit]+=1
    @staticmethod
    def iterateHashtags(timeRange, folderType):
        for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(folderType,'%s_%s'%timeRange)): 
            hashtagObject = Hashtag(h)
            if hashtagObject.isValidObject(): yield hashtagObject

class Simulation:
    TIME_WINDOW_IN_SECONDS = 5*60
    @staticmethod
    def runModel(latticeSelectionModel, latticeGraph, hashtagsIterator):
        currentLattices = latticeGraph.nodes()
        for hashtag in hashtagsIterator:
            for timeUnit, occs in enumerate(hashtag.getOccrancesEveryTimeWindowIterator(Simulation.TIME_WINDOW_IN_SECONDS)):
                hashtag.updateOccuranceDistributionInLattices(timeUnit, occs)
                hashtag.updateOccurancesInTargetLattices(timeUnit, hashtag.occuranceDistributionInLattices)
                latticeSelectionModel.selectNextLatticesRandomly(timeUnit, hashtag, latticeGraph, hashtag.occuranceDistributionInLattices)
            for k,method in EvaluationMetrics.iteritems(): print k, method(hashtag.occuranceDistributionInLattices, hashtag.targetLattices )
            exit()
    @staticmethod
    def run():
        timeRange, folderType = (2,11), 'world'
        graph = LatticeGraph(hashtagsLatticeGraphFile%(folderType,'%s_%s'%timeRange), LatticeGraph.typeSharingProbability).load()
        Simulation.runModel(LatticeSelectionModel(), graph, Hashtag.iterateHashtags(timeRange, folderType))

if __name__ == '__main__':
    Simulation.run()