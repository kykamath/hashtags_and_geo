'''
Created on Dec 8, 2011

@author: kykamath
'''
import sys
sys.path.append('../')
from library.file_io import FileIO
from settings import hashtagsWithoutEndingWindowFile, hashtagsLatticeGraphFile,\
    hashtagsFile, hashtagsModelsFolder
from experiments.mr_area_analysis import getOccuranesInHighestActiveRegion,\
    TIME_UNIT_IN_SECONDS, LATTICE_ACCURACY, HashtagsClassifier,\
    getOccurranceDistributionInEpochs,\
    getRadius
import numpy as np
from library.stats import getOutliersRangeUsingIRQ
from library.geo import getHaversineDistanceForLids, getLatticeLid
from collections import defaultdict
from operator import itemgetter
import networkx as nx
from library.graphs import plot
import datetime, math, random
from library.classes import GeneralMethods
import matplotlib.pyplot as plt
from library.plotting import plot3D

def filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeHashtags, neighborHashtags, findLag=True):
    if findLag: 
        dataToReturn = [(hashtag, np.abs(latticeHashtags[hashtag][0]-timeTuple[0])/TIME_UNIT_IN_SECONDS) for hashtag, timeTuple in neighborHashtags.iteritems() if hashtag in latticeHashtags]
        _, upperRangeForTemporalDistance = getOutliersRangeUsingIRQ(zip(*(dataToReturn))[1])
        return dict(filter(lambda t: t[1]<=upperRangeForTemporalDistance, dataToReturn))
    else: 
        dataToReturn = [(hashtag, timeTuple, np.abs(latticeHashtags[hashtag][0]-timeTuple[0])/TIME_UNIT_IN_SECONDS) for hashtag, timeTuple in neighborHashtags.iteritems() if hashtag in latticeHashtags]
        _, upperRangeForTemporalDistance = getOutliersRangeUsingIRQ(zip(*(dataToReturn))[2])
        return dict([(t[0], t[1]) for t in dataToReturn if t[2]<=upperRangeForTemporalDistance])
#def latticeNodeByHaversineDistance(latticeObject):
#    dataToReturn = {'id': latticeObject['id'], 'links': {}}
#    for neighborLattice, _ in latticeObject['links'].iteritems(): dataToReturn['links'][neighborLattice]=getHaversineDistanceForLids(latticeObject['id'].replace('_', ' '), neighborLattice.replace('_', ' '))
#    return dataToReturn
#def latticeNodeBySharingProbability(latticeObject):
#    dataToReturn = {'id': latticeObject['id'], 'links': {}}
#    latticeHashtagsSet = set(latticeObject['hashtags'])
#    for neighborLattice, neighborHashtags in latticeObject['links'].iteritems():
#        neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags)
#        neighborHashtagsSet = set(neighborHashtags)
#        dataToReturn['links'][neighborLattice]=len(latticeHashtagsSet.intersection(neighborHashtagsSet))/float(len(latticeHashtagsSet))
#    return dataToReturn
#def latticeNodeByTemporalDistanceInHours(latticeObject):
#    dataToReturn = {'id': latticeObject['id'], 'links': {}}
#    for neighborLattice, neighborHashtags in latticeObject['links'].iteritems():
#        neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags)
#        dataX = zip(*neighborHashtags.iteritems())[1]
#        dataToReturn['links'][neighborLattice]=np.mean(dataX)
#    return dataToReturn
#def latticeNodeByTemporalClosenessScore(latticeObject):
#    dataToReturn = {'id': latticeObject['id'], 'links': {}}
#    for neighborLattice, neighborHashtags in latticeObject['links'].iteritems():
#        neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags)
#        dataX = map(lambda lag: LatticeGraph.temporalScore(lag, LatticeGraph.upperRangeForTemporalDistances), zip(*neighborHashtags.iteritems())[1])
#        dataToReturn['links'][neighborLattice]=np.mean(dataX)
#    return dataToReturn
#def latticeNodeByHashtagDiffusionLocationVisitation(latticeObject, generateTemporalClosenessScore=False):
#    def updateHashtagDistribution(latticeObj, hashtagDistribution): 
#        for h, (t, _) in latticeObj['hashtags'].iteritems(): hashtagDistribution[h].append([latticeObj['id'], t])
#    dataToReturn = {'id': latticeObject['id'], 'links': {}}
#    hashtagDistribution, numberOfHastagsAtCurrentLattice = defaultdict(list), float(len(latticeObject['hashtags']))
#    updateHashtagDistribution(latticeObject, hashtagDistribution)
#    for neighborLattice, neighborHashtags in latticeObject['links'].iteritems(): 
#        neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags, findLag=False)
#        updateHashtagDistribution({'id':neighborLattice, 'hashtags': neighborHashtags}, hashtagDistribution)
#    latticeEdges = {'in': defaultdict(float), 'out': defaultdict(float)}
#    for hashtag in hashtagDistribution.keys()[:]: 
#        hashtagOccurences = sorted(hashtagDistribution[hashtag], key=itemgetter(1))
#        for neighborLattice, lag in [(t[0], (t[1]-latticeObject['hashtags'][hashtag][0])/TIME_UNIT_IN_SECONDS) for t in hashtagOccurences if t[0]!=latticeObject['id']]:
#            if not generateTemporalClosenessScore:
#                if lag<0: latticeEdges['in'][neighborLattice]+=-lag/numberOfHastagsAtCurrentLattice
#                else: latticeEdges['out'][neighborLattice]+=lag/numberOfHastagsAtCurrentLattice
#            else: 
#                if lag<0: latticeEdges['in'][neighborLattice]+=LatticeGraph.temporalScore(-lag, LatticeGraph.upperRangeForTemporalDistances)/numberOfHastagsAtCurrentLattice
#                else: latticeEdges['out'][neighborLattice]+=LatticeGraph.temporalScore(lag, LatticeGraph.upperRangeForTemporalDistances)/numberOfHastagsAtCurrentLattice
#    dataToReturn['links']=latticeEdges
#    return dataToReturn
#class LatticeGraph:
#    upperRangeForTemporalDistances = 8.24972222222
#    typeSharingProbability = {'id': 'sharing_probability', 'method': latticeNodeBySharingProbability, 'title': 'Probability of sharing hastags'}
#    typeTemporalCloseness = {'id': 'temporal_closeness', 'method': latticeNodeByTemporalClosenessScore, 'title': 'Temporal closeness'}
#    typeTemporalDistanceInHours = {'id': 'temporal_distance_in_hours', 'method': latticeNodeByTemporalDistanceInHours, 'title': 'Temporal distance (hours)'}
#    typeHaversineDistance = {'id': 'haversine_distance', 'method': latticeNodeByHaversineDistance, 'title': 'Distance (miles)'}
#    def __init__(self, graphFile, latticeGraphType, graphType=nx.Graph):
#        self.graphFile = graphFile
#        self.latticeGraphType = latticeGraphType
##        self.graphType = graphType
#    def load(self):
#        i = 1
#        self.graph = nx.DiGraph()
#        for latticeObject in FileIO.iterateJsonFromFile(self.graphFile):
#            print i; i+=1
#            latticeObject = self.latticeGraphType['method'](latticeObject)
#            if 'in' in latticeObject['links']:
##                for no, w in latticeObject['links']['in'].iteritems(): self.graph.add_edge(no, latticeObject['id'], {'w': w})
##                for no, w in latticeObject['links']['out'].iteritems(): self.graph.add_edge(latticeObject['id'], no, {'w': w})
#                no, w = max(latticeObject['links']['in'].iteritems(), key=itemgetter(1))
#                self.graph.add_edge(no, latticeObject['id'], {'w': w})
#                no, w = max(latticeObject['links']['out'].iteritems(), key=itemgetter(1))
#                self.graph.add_edge(latticeObject['id'], no, {'w': w})
#            else:
##                for no, w in latticeObject['links'].iteritems(): self.graph.add_edge(latticeObject['id'], no, {'w': w}), self.graph.add_edge(no, latticeObject['id'], {'w': w})
#                no, w = max(latticeObject['links'].iteritems(), key=itemgetter(1))
#                if not self.graph.has_edge(latticeObject['id'], no) or self.graph.edge[latticeObject['id']][no]['w']<w: 
#                    self.graph.add_edge(latticeObject['id'], no, {'w': w}), self.graph.add_edge(no, latticeObject['id'], {'w': w})
#            if i==100: break
##        plot(self.graph)
#        return self.graph
#    @staticmethod
#    def normalizeNode(latticeNodeObject):
#        if 'in' in latticeNodeObject['links']:
#            for edgeType, edges in latticeNodeObject['links'].iteritems():
#                totalEdgeWeight = sum(edges.values())
#                for lattice, score in edges.items()[:]: latticeNodeObject['links'][edgeType][lattice] = score/totalEdgeWeight
#        else:
#            totalEdgeWeight = sum(latticeNodeObject['links'].values())
#            for lattice, score in latticeNodeObject['links'].items()[:]: latticeNodeObject['links'][lattice]=score/totalEdgeWeight
#    @staticmethod
#    def temporalScore(lag, width):
#        lag=int(lag*TIME_UNIT_IN_SECONDS)   
#        width=int(width*TIME_UNIT_IN_SECONDS)
#        if lag==0: return 1.0
#        elif lag>=width: return 0.0
#        return 1-np.log(lag)/np.log(width)
#    @staticmethod
#    def run(timeRange, folderType):
#        latticeGraph = LatticeGraph(hashtagsLatticeGraphFile%(folderType,'%s_%s'%timeRange), LatticeGraph.typeTemporalCloseness)
#        latticeGraph.load()
#        print latticeGraph.graph.number_of_nodes()

GREEDY_LATTICE_SELECTION_MODEL = 'greedy'
SHARING_PROBABILITY_LATTICE_SELECTION_MODEL = 'sharing_probability'

class Metrics:
    @staticmethod
    def overallOccurancesHitRate(hashtag):
        totalOccurances, occurancesObserved = 0., 0.
        if hashtag.occuranceDistributionInTargetLattices:
            for k,v in hashtag.occuranceDistributionInLattices.iteritems(): totalOccurances+=len(v)
            for k, v in hashtag.occuranceDistributionInTargetLattices.iteritems(): occurancesObserved+=sum(v['occurances'].values())
            return occurancesObserved/totalOccurances
    @staticmethod
    def occurancesHitRateAfterTargetSelection(hashtag):
        totalOccurances, occurancesObserved = 0., 0.
        if hashtag.occuranceDistributionInTargetLattices:
            targetSelectionTimeUnit = min(v['selectedTimeUnit'] for v in hashtag.occuranceDistributionInTargetLattices.values())
            for k,v in hashtag.occuranceDistributionInLattices.iteritems(): totalOccurances+=len([i for i in v if i>targetSelectionTimeUnit])
            for k, v in hashtag.occuranceDistributionInTargetLattices.iteritems(): occurancesObserved+=sum(v['occurances'].values())
            if totalOccurances!=0.: return occurancesObserved/totalOccurances
            return None
    @staticmethod
    def occurancesMissRateBeforeTargetSelection(hashtag):
        totalOccurances, occurancesBeforeTimeUnit = 0., 0.
        if hashtag.occuranceDistributionInTargetLattices:
            targetSelectionTimeUnit = min(v['selectedTimeUnit'] for v in hashtag.occuranceDistributionInTargetLattices.values())
            for k,v in hashtag.occuranceDistributionInLattices.iteritems(): totalOccurances+=len(v); occurancesBeforeTimeUnit+=len([i for i in v if i<=targetSelectionTimeUnit])
            return occurancesBeforeTimeUnit/totalOccurances
EvaluationMetrics = {
                     'overall_hit_rate': Metrics.overallOccurancesHitRate,
                     'hit_rate_after_target_selection': Metrics.occurancesHitRateAfterTargetSelection,
                     'miss_rate_before_target_selection': Metrics.occurancesMissRateBeforeTargetSelection
                     }

class LatticeSelectionModel(object):
    def __init__(self, id='random', **kwargs):
        self.id = id
        self.params = kwargs['params']
        self.budget = self.params.get('budget', None)
        self.trainingHashtagsFile = kwargs.get('trainingHashtagsFile', None)
        self.testingHashtagsFile = kwargs.get('testingHashtagsFile', None)
        self.evaluationName = kwargs.get('evaluationName', '')
    def selectTargetLattices(self, currentTimeUnit, hashtag): return random.sample(hashtag.occuranceDistributionInLattices, min([self.budget, len(hashtag.occuranceDistributionInLattices)]))
    def getModelSimulationFile(self): 
        file = hashtagsModelsFolder%('world', self.id)+'%s.eva'%self.params['evaluationName']; FileIO.createDirectoryForFile(file); return file
    def evaluateModel(self):
        hashtags = {}
        for h in FileIO.iterateJsonFromFile(self.testingHashtagsFile): 
            hashtag = Hashtag(h)
            if hashtag.isValidObject():
                for timeUnit, occs in enumerate(hashtag.getOccrancesEveryTimeWindowIterator(HashtagsClassifier.CLASSIFIER_TIME_UNIT_IN_SECONDS)):
                    hashtag.updateOccuranceDistributionInLattices(timeUnit, occs)
                    hashtag.updateOccurancesInTargetLattices(timeUnit, hashtag.occuranceDistributionInLattices)
                    if self.params['timeUnitToPickTargetLattices']==timeUnit: hashtag._initializeTargetLattices(timeUnit, self.selectTargetLattices(timeUnit, hashtag))
                hashtags[hashtag.hashtagObject['h']] = {'model': self.id, 'classId': hashtag.hashtagClassId, 'metrics': dict([(k, method(hashtag))for k,method in EvaluationMetrics.iteritems()])}
        return hashtags
    def evaluateModelWithVaryingTimeUnitToPickTargetLattices(self, numberOfTimeUnits = 24):
        self.params['evaluationName'] = 'time'
        GeneralMethods.runCommand('rm -rf %s'%self.getModelSimulationFile())
        for t in range(numberOfTimeUnits):
            print 'Evaluating at t=%d'%t, self.getModelSimulationFile()
            self.params['timeUnitToPickTargetLattices'] = t
            FileIO.writeToFileAsJson({'params': self.params, 'hashtags': self.evaluateModel()}, self.getModelSimulationFile())
    def evaluateModelWithVaryingBudget(self, budgetLimit = 20):
        self.params['evaluationName'] = 'budget'
        GeneralMethods.runCommand('rm -rf %s'%self.getModelSimulationFile())
        for b in range(1, budgetLimit):
            print 'Evaluating at budget=%d'%b, self.getModelSimulationFile()
            self.params['budget'] = b
            FileIO.writeToFileAsJson({'params': self.params, 'hashtags': self.evaluateModel()}, self.getModelSimulationFile())
    def evaluateByVaringBudgetAndTimeUnits(self, numberOfTimeUnits=24, budgetLimit = 20):
        self.params['evaluationName'] = 'budget_time'
        GeneralMethods.runCommand('rm -rf %s'%self.getModelSimulationFile())
        for b in range(1, budgetLimit):
            for t in range(numberOfTimeUnits):
                print 'Evaluating at budget=%d, numberOfTimeUnits=%d'%(b, t), self.getModelSimulationFile()
                self.params['budget'] = b
                self.params['timeUnitToPickTargetLattices'] = t
                FileIO.writeToFileAsJson({'params': self.params, 'hashtags': self.evaluateModel()}, self.getModelSimulationFile())
    def plotModelWithVaryingTimeUnitToPickTargetLattices(self):
        self.params['evaluationName'] = 'time'
        metricDistributionInTimeUnits = defaultdict(dict)
        for data in FileIO.iterateJsonFromFile(self.getModelSimulationFile()):
            t = data['params']['timeUnitToPickTargetLattices']
            for h in data['hashtags']:
#                if data['hashtags'][h]['classId']==3:
                    for metric in data['hashtags'][h]['metrics']:
                        if metric not in metricDistributionInTimeUnits: metricDistributionInTimeUnits[metric] = defaultdict(list)
                        metricDistributionInTimeUnits[metric][t].append(data['hashtags'][h]['metrics'][metric])
        for metric, metricValues in metricDistributionInTimeUnits.iteritems():
            dataX, dataY = zip(*[(t, np.mean(filter(lambda v: v!=None, values))) for i, (t, values) in enumerate(metricValues.iteritems())])
            plt.plot(dataX, dataY, label=metric)
        plt.legend()
        plt.show()
    def plotModelWithVaryingBudget(self):
        self.params['evaluationName'] = 'budget'
        metricDistributionInTimeUnits = defaultdict(dict)
        for data in FileIO.iterateJsonFromFile(self.getModelSimulationFile()):
            b = data['params']['budget']
            for h in data['hashtags']:
#                if data['hashtags'][h]['classId']==3:
                    for metric in data['hashtags'][h]['metrics']:
                        if metric not in metricDistributionInTimeUnits: metricDistributionInTimeUnits[metric] = defaultdict(list)
                        metricDistributionInTimeUnits[metric][b].append(data['hashtags'][h]['metrics'][metric])
        for metric, metricValues in metricDistributionInTimeUnits.iteritems():
            dataX, dataY = zip(*[(t, np.mean(filter(lambda v: v!=None, values))) for i, (t, values) in enumerate(metricValues.iteritems())])
            plt.plot(dataX, dataY, label=metric)
        plt.legend()
        plt.show()
    def plotVaringBudgetAndTimeUnits(self):
        # overall_hit_rate, miss_rate_before_target_selection, hit_rate_after_target_selection
        metrics = ['hit_rate_after_target_selection']
        for metric in metrics:
            self.params['evaluationName'] = 'budget_time'
            scoreDistribution = defaultdict(dict)
    #        print self.getModelSimulationFile()
            for data in FileIO.iterateJsonFromFile(self.getModelSimulationFile()):
                timeUnit, budget = data['params']['timeUnitToPickTargetLattices'], data['params']['budget']
                for h in data['hashtags']: 
                    if data['hashtags'][h]['metrics'][metric]!=None: 
                        if budget not in scoreDistribution[timeUnit]: scoreDistribution[timeUnit][budget] = []
                        scoreDistribution[timeUnit][budget].append(data['hashtags'][h]['metrics'][metric])
            for timeUnit in scoreDistribution:
                for budget in scoreDistribution[timeUnit]:
                    scoreDistribution[timeUnit][budget] = np.mean(scoreDistribution[timeUnit][budget])
            print metric
            plot3D(scoreDistribution)
            plt.show()
            
class GreedyLatticeSelectionModel(LatticeSelectionModel):
    ''' Pick the location with maximum observations till that time.
    '''
    def __init__(self, **kwargs): super(GreedyLatticeSelectionModel, self).__init__(GREEDY_LATTICE_SELECTION_MODEL, **kwargs)
    def selectTargetLattices(self, currentTimeUnit, hashtag):return zip(*sorted(hashtag.occuranceDistributionInLattices.iteritems(), key=lambda t: len(t), reverse=True))[0][:self.params['budget']]

class SharingProbabilityLatticeSelectionModel(LatticeSelectionModel):
    ''' Pick the location with highest probability:
        score(l1) = P(l2)*P(l1|l2) + P(l3)*P(l1|l3) ... (where l2 and l3 are observed).
    '''
    def __init__(self, folderType, timeRange, **kwargs): 
        super(SharingProbabilityLatticeSelectionModel, self).__init__(SHARING_PROBABILITY_LATTICE_SELECTION_MODEL, **kwargs)
        self.graphFile = hashtagsLatticeGraphFile%(folderType,'%s_%s'%timeRange)
        self.initializeModel()
    def initializeModel(self):
        self.model = {'sharingProbaility': defaultdict(dict), 'hashtagObservingProbability': {}}
        hashtagsObserved = []
        for latticeObject in FileIO.iterateJsonFromFile(self.graphFile):
            latticeHashtagsSet = set(latticeObject['hashtags'])
            hashtagsObserved+=latticeObject['hashtags']
            self.model['hashtagObservingProbability'][latticeObject['id']] = latticeHashtagsSet
            for neighborLattice, neighborHashtags in latticeObject['links'].iteritems():
                neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags)
                neighborHashtagsSet = set(neighborHashtags)
                self.model['sharingProbaility'][latticeObject['id']][neighborLattice]=len(latticeHashtagsSet.intersection(neighborHashtagsSet))/float(len(latticeHashtagsSet))
            self.model['sharingProbaility'][latticeObject['id']][latticeObject['id']]=1.0
        totalNumberOfHashtagsObserved=float(len(set(hashtagsObserved)))
        for lattice in self.model['hashtagObservingProbability'].keys()[:]: self.model['hashtagObservingProbability'][lattice] = len(self.model['hashtagObservingProbability'][lattice])/totalNumberOfHashtagsObserved
    def selectTargetLattices(self, currentTimeUnit, hashtag): 
        latticeScores = defaultdict(float)
        for currentLattice in hashtag.occuranceDistributionInLattices:
            for neighborLattice in self.model['sharingProbaility'][currentLattice]: latticeScores[neighborLattice]+=math.log(self.model['hashtagObservingProbability'][currentLattice])+math.log(self.model['sharingProbaility'][currentLattice][neighborLattice])
        if latticeScores: return zip(*sorted(latticeScores.iteritems(), key=lambda t: itemgetter(1), reverse=True))[0][:self.params['budget']]
        else: return hashtag.occuranceDistributionInLattices.keys()[:self.params['budget']]
            
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
        self.occuranceDistributionInTargetLattices = {}
        if self.hashtagObject['oc']: 
            self.timePeriod = (self.hashtagObject['oc'][-1][1]-self.hashtagObject['oc'][0][1])/TIME_UNIT_IN_SECONDS
            self.hashtagClassId = HashtagsClassifier.classify(self.hashtagObject)
        # Data structures for building classifier.
        if dataStructuresToBuildClassifier and self.isValidObject():
            self.classifiable = True
            occurranceDistributionInEpochs = getOccurranceDistributionInEpochs(getOccuranesInHighestActiveRegion(self.hashtagObject), timeUnit=HashtagsClassifier.CLASSIFIER_TIME_UNIT_IN_SECONDS, fillInGaps=True, occurancesCount=False)
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
        if self.hashtagClassId==None: return False
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
            self.occuranceDistributionInTargetLattices[lattice] = {'selectedTimeUnit': None,'occurances':defaultdict(int)}
            self.occuranceDistributionInTargetLattices[lattice]['selectedTimeUnit'] = currentTimeUnit 
    def updateOccurancesInTargetLattices(self, currentTimeUnit, occuranceDistributionInLattices):
        for targetLattice in self.occuranceDistributionInTargetLattices:
            for occuranceTimeUnit in occuranceDistributionInLattices[targetLattice]: 
                if occuranceTimeUnit==currentTimeUnit: 
                    self.occuranceDistributionInTargetLattices[targetLattice]['occurances'][currentTimeUnit]+=1
#    @staticmethod
#    def iterateHashtags(timeRange, folderType):
#        for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(folderType,'%s_%s'%timeRange)): 
#            hashtagObject = Hashtag(h)
#            if hashtagObject.isValidObject(): yield hashtagObject

class Simulation:
    trainingHashtagsFile = hashtagsFile%('training_world','%s_%s'%(2,11))
    testingHashtagsFile = hashtagsFile%('testing_world','%s_%s'%(2,11))
    @staticmethod
    def varyingTimeUnitToPickTargetLattices():
        params = dict(budget=100, timeUnitToPickTargetLattices=6)
        SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingTimeUnitToPickTargetLattices(numberOfTimeUnits=24)
#        SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).plotModelWithVaryingTimeUnitToPickTargetLattices()
#        GreedyLatticeSelectionModel(params=params, testingHashtagsFile=Simulation.testingHashtagsFile).evaluateModelWithVaryingTimeUnitToPickTargetLattices(numberOfTimeUnits=24)
#        GreedyLatticeSelectionModel(params=params, testingHashtagsFile=Simulation.testingHashtagsFile).plotModelWithVaryingTimeUnitToPickTargetLattices()
    @staticmethod
    def varyingBudgetToPickTargetLattices():
        params = dict(budget=100, timeUnitToPickTargetLattices=6)
#        SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingBudget()
#        SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).plotModelWithVaryingBudget()
#        GreedyLatticeSelectionModel(params=params, testingHashtagsFile=Simulation.testingHashtagsFile).evaluateModelWithVaryingBudget(budgetLimit=20)
        GreedyLatticeSelectionModel(params=params, testingHashtagsFile=Simulation.testingHashtagsFile).plotModelWithVaryingBudget()
    @staticmethod
    def varyingBudgetAndTime():
#        SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params={}).evaluateByVaringBudgetAndTimeUnits()
        GreedyLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params={}).plotVaringBudgetAndTimeUnits()
        
if __name__ == '__main__':
    Simulation.varyingTimeUnitToPickTargetLattices()
#    SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), params={})
#    model.saveModelSimulation()