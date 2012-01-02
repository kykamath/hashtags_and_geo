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

GREEDY_LATTICE_SELECTION_MODEL = 'greedy'
SHARING_PROBABILITY_LATTICE_SELECTION_MODEL = 'sharing_probability'
TRANSMITTING_PROBABILITY_LATTICE_SELECTION_MODEL = 'transmitting_probability'

class Metrics:
    overall_hit_rate = 'overall_hit_rate'
    hit_rate_after_target_selection = 'hit_rate_after_target_selection'
    miss_rate_before_target_selection = 'miss_rate_before_target_selection'
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
                     Metrics.overall_hit_rate: Metrics.overallOccurancesHitRate,
                     Metrics.hit_rate_after_target_selection: Metrics.occurancesHitRateAfterTargetSelection,
                     Metrics.miss_rate_before_target_selection: Metrics.occurancesMissRateBeforeTargetSelection
                     }

class LatticeSelectionModel(object):
    def __init__(self, id='random', **kwargs):
        self.id = id
        self.params = kwargs['params']
        self.budget = self.params.get('budget', None)
        self.trainingHashtagsFile = kwargs.get('trainingHashtagsFile', None)
        self.testingHashtagsFile = kwargs.get('testingHashtagsFile', None)
        self.evaluationName = kwargs.get('evaluationName', '')
    def selectTargetLattices(self, currentTimeUnit, hashtag): 
        ''' Pick random lattices observed till now.
        '''
        return random.sample(hashtag.occuranceDistributionInLattices, min([self.budget, len(hashtag.occuranceDistributionInLattices)]))
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
    @staticmethod
    def plotModelWithVaryingTimeUnitToPickTargetLattices(models, metric, **kwargs):
        for model in models:
            model = model(**kwargs)
            model.params['evaluationName'] = 'time'
            metricDistributionInTimeUnits = defaultdict(dict)
            for data in FileIO.iterateJsonFromFile(model.getModelSimulationFile()):
                t = data['params']['timeUnitToPickTargetLattices']
                for h in data['hashtags']:
    #                if data['hashtags'][h]['classId']==3:
#                        for metric in metric:
                        if metric not in metricDistributionInTimeUnits: metricDistributionInTimeUnits[metric] = defaultdict(list)
                        metricDistributionInTimeUnits[metric][t].append(data['hashtags'][h]['metrics'][metric])
            for metric, metricValues in metricDistributionInTimeUnits.iteritems():
                dataX, dataY = zip(*[(t, np.mean(filter(lambda v: v!=None, values))) for i, (t, values) in enumerate(metricValues.iteritems())])
                plt.plot(dataX, dataY, label=model.id)
        plt.legend(loc=4)
        plt.title('%s comparison'%metric)
        plt.show()
    @staticmethod
    def plotModelWithVaryingBudget(models, metric, **kwargs):
        for model in models:
            model = model(**kwargs)
            model.params['evaluationName'] = 'budget'
            metricDistributionInTimeUnits = defaultdict(dict)
            for data in FileIO.iterateJsonFromFile(model.getModelSimulationFile()):
                t = data['params']['budget']
                for h in data['hashtags']:
    #                if data['hashtags'][h]['classId']==3:
#                        for metric in metric:
                        if metric not in metricDistributionInTimeUnits: metricDistributionInTimeUnits[metric] = defaultdict(list)
                        metricDistributionInTimeUnits[metric][t].append(data['hashtags'][h]['metrics'][metric])
            for metric, metricValues in metricDistributionInTimeUnits.iteritems():
                dataX, dataY = zip(*[(t, np.mean(filter(lambda v: v!=None, values))) for i, (t, values) in enumerate(metricValues.iteritems())])
                plt.plot(dataX, dataY, label=model.id)
        plt.legend(loc=4)
        plt.title('%s comparison'%metric)
        plt.show()
    def plotVaringBudgetAndTimeUnits(self):
        # overall_hit_rate, miss_rate_before_target_selection, hit_rate_after_target_selection
        metrics = [Metrics.overall_hit_rate]
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
    def selectTargetLattices(self, currentTimeUnit, hashtag): return zip(*sorted(hashtag.occuranceDistributionInLattices.iteritems(), key=lambda t: len(t[1]), reverse=True))[0][:self.params['budget']]

class SharingProbabilityLatticeSelectionModel(LatticeSelectionModel):
    ''' Pick the location with highest probability:
        score(l1) = P(l2)*P(l1|l2) + P(l3)*P(l1|l3) ... (where l2 and l3 are observed).
    '''
    def __init__(self, folderType=None, timeRange=None, **kwargs): 
        super(SharingProbabilityLatticeSelectionModel, self).__init__(SHARING_PROBABILITY_LATTICE_SELECTION_MODEL, **kwargs)
        if folderType:
            self.graphFile = hashtagsLatticeGraphFile%(folderType,'%s_%s'%timeRange)
            self.initializeModel()
    def initializeModel(self):
        self.model = {'neighborProbability': defaultdict(dict), 'hashtagObservingProbability': {}}
        hashtagsObserved = []
        for latticeObject in FileIO.iterateJsonFromFile(self.graphFile):
            latticeHashtagsSet = set(latticeObject['hashtags'])
            hashtagsObserved+=latticeObject['hashtags']
            self.model['hashtagObservingProbability'][latticeObject['id']] = latticeHashtagsSet
            for neighborLattice, neighborHashtags in latticeObject['links'].iteritems():
                neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags)
                neighborHashtagsSet = set(neighborHashtags)
                self.model['neighborProbability'][latticeObject['id']][neighborLattice]=len(latticeHashtagsSet.intersection(neighborHashtagsSet))/float(len(latticeHashtagsSet))
            self.model['neighborProbability'][latticeObject['id']][latticeObject['id']]=1.0
        totalNumberOfHashtagsObserved=float(len(set(hashtagsObserved)))
        for lattice in self.model['hashtagObservingProbability'].keys()[:]: self.model['hashtagObservingProbability'][lattice] = len(self.model['hashtagObservingProbability'][lattice])/totalNumberOfHashtagsObserved
    def selectTargetLattices(self, currentTimeUnit, hashtag): 
        targetLattices = zip(*sorted(hashtag.occuranceDistributionInLattices.iteritems(), key=lambda t: len(t[1]), reverse=True))[0][:self.params['budget']]
        targetLattices = list(targetLattices)
        if len(targetLattices)<self.params['budget']: 
            latticeScores = defaultdict(float)
            for currentLattice in hashtag.occuranceDistributionInLattices:
                for neighborLattice in self.model['neighborProbability'][currentLattice]: latticeScores[neighborLattice]+=math.log(self.model['hashtagObservingProbability'][currentLattice])+math.log(self.model['neighborProbability'][currentLattice][neighborLattice])
#                for lattice in latticeScores:
#                    noOfOccurances = len(hashtag.occuranceDistributionInLattices.get(lattice, []))
#                    if noOfOccurances!=0: latticeScores[lattice]+=math.log(noOfOccurances)
                extraTargetLattices = sorted(latticeScores.iteritems(), key=itemgetter(1))
#                extraTargetLattices.reverse()
                while len(targetLattices)<self.params['budget'] and extraTargetLattices:
                    t = extraTargetLattices.pop()
                    if t[0] not in targetLattices: targetLattices.append(t[0])
        assert len(targetLattices)<=self.params['budget']
        return targetLattices
    
class TransmittingProbabilityLatticeSelectionModel(SharingProbabilityLatticeSelectionModel):
    def __init__(self, folderType=None, timeRange=None, **kwargs): 
        super(TransmittingProbabilityLatticeSelectionModel, self).__init__(TRANSMITTING_PROBABILITY_LATTICE_SELECTION_MODEL, folderType, timeRange, **kwargs)
    def initializeModel(self):
        self.model = {'neighborProbability': defaultdict(dict), 'hashtagObservingProbability': {}}
        hashtagsObserved = []
        for latticeObject in FileIO.iterateJsonFromFile(self.graphFile):
            latticeHashtagsSet = set(latticeObject['hashtags'])
            hashtagsObserved+=latticeObject['hashtags']
            self.model['hashtagObservingProbability'][latticeObject['id']] = latticeHashtagsSet
            for neighborLattice, neighborHashtags in latticeObject['links'].iteritems():
                neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags)
                neighborHashtagsSet = set(neighborHashtags)
                self.model['neighborProbability'][latticeObject['id']][neighborLattice]=len(latticeHashtagsSet.intersection(neighborHashtagsSet))/float(len(latticeHashtagsSet))
            self.model['neighborProbability'][latticeObject['id']][latticeObject['id']]=1.0
        totalNumberOfHashtagsObserved=float(len(set(hashtagsObserved)))
        for lattice in self.model['hashtagObservingProbability'].keys()[:]: self.model['hashtagObservingProbability'][lattice] = len(self.model['hashtagObservingProbability'][lattice])/totalNumberOfHashtagsObserved
        
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
    def run():
        params = dict(budget=5, timeUnitToPickTargetLattices=6)
#        SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingTimeUnitToPickTargetLattices()
#        SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingBudget()
#        SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateByVaringBudgetAndTimeUnits()
#        LatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingTimeUnitToPickTargetLattices()
#        LatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingBudget()
#        LatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateByVaringBudgetAndTimeUnits()
#        GreedyLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingTimeUnitToPickTargetLattices()
#        GreedyLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingBudget()
#        GreedyLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateByVaringBudgetAndTimeUnits()
        LatticeSelectionModel.plotModelWithVaryingBudget([LatticeSelectionModel, SharingProbabilityLatticeSelectionModel,
                                                                                GreedyLatticeSelectionModel], 
                                                                               Metrics.overall_hit_rate, 
                                                                                   params=params)
#        SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).plotVaringBudgetAndTimeUnits()
        
if __name__ == '__main__':
    Simulation.run()
#    SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), params={})
#    model.saveModelSimulation()