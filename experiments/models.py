'''
Created on Dec 8, 2011

@author: kykamath
'''
import sys
sys.path.append('../')
from library.file_io import FileIO
from settings import hashtagsWithoutEndingWindowFile, hashtagsLatticeGraphFile,\
    hashtagsFile, hashtagsModelsFolder, hashtagsAnalysisFolder,\
    hashtagsClassifiersFolder
from experiments.mr_area_analysis import getOccuranesInHighestActiveRegion,\
    TIME_UNIT_IN_SECONDS, LATTICE_ACCURACY, HashtagsClassifier,\
    getOccurranceDistributionInEpochs,\
    getRadius
import numpy as np
from library.stats import getOutliersRangeUsingIRQ
from library.geo import getHaversineDistanceForLids, getLatticeLid, getLocationFromLid
from collections import defaultdict
from operator import itemgetter
import networkx as nx
from library.graphs import plot
import datetime, math, random
from library.classes import GeneralMethods
import matplotlib.pyplot as plt
from library.plotting import plot3D
from sklearn.svm import SVC
from sklearn.externals import joblib

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
SHARING_PROBABILITY_LATTICE_SELECTION_WITH_LOCALITY_CLASSIFIER_MODEL = 'sharing_probability_with_locality_classifier'

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
            hashtag = Hashtag(h, dataStructuresToBuildClassifier=self.params.get('dataStructuresToBuildClassifier', False))
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
    def __init__(self, id=SHARING_PROBABILITY_LATTICE_SELECTION_MODEL, folderType=None, timeRange=None, **kwargs): 
        super(SharingProbabilityLatticeSelectionModel, self).__init__(id, **kwargs)
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
                for neighborLattice in self.model['neighborProbability'][currentLattice]: 
                    if self.model['neighborProbability'][currentLattice][neighborLattice] > 0: latticeScores[neighborLattice]+=math.log(self.model['hashtagObservingProbability'][currentLattice])+math.log(self.model['neighborProbability'][currentLattice][neighborLattice])
            extraTargetLattices = sorted(latticeScores.iteritems(), key=itemgetter(1))
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
                neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags, findLag=False)
#                neighborHashtagsSet = set(neighborHashtags)
                transmittedHashtags = [k for k in neighborHashtags if k in latticeObject['hashtags'] and latticeObject['hashtags'][k][0]<neighborHashtags[k][0]]
                self.model['neighborProbability'][latticeObject['id']][neighborLattice]=len(transmittedHashtags)/float(len(latticeHashtagsSet))
            self.model['neighborProbability'][latticeObject['id']][latticeObject['id']]=1.0
        totalNumberOfHashtagsObserved=float(len(set(hashtagsObserved)))
        for lattice in self.model['hashtagObservingProbability'].keys()[:]: self.model['hashtagObservingProbability'][lattice] = len(self.model['hashtagObservingProbability'][lattice])/totalNumberOfHashtagsObserved
        
class SharingProbabilityLatticeSelectionWithLocalityClassifierModel(SharingProbabilityLatticeSelectionModel):
    def __init__(self, folderType=None, timeRange=None, **kwargs): 
        super(SharingProbabilityLatticeSelectionWithLocalityClassifierModel, self).__init__(SHARING_PROBABILITY_LATTICE_SELECTION_WITH_LOCALITY_CLASSIFIER_MODEL, folderType, timeRange, **kwargs)
    def selectTargetLattices(self, currentTimeUnit, hashtag): 
        classifier = LocalityClassifier(currentTimeUnit+1, features=LocalityClassifier.FEATURES_AGGGREGATED_OCCURANCES_RADIUS)
        localityClassId = classifier.predict(hashtag)
        targetLattices = zip(*sorted(hashtag.occuranceDistributionInLattices.iteritems(), key=lambda t: len(t[1]), reverse=True))[0][:self.params['budget']]
        targetLattices = list(targetLattices)
        if len(targetLattices)<self.params['budget']: 
            latticeScores = defaultdict(float)
            for currentLattice in hashtag.occuranceDistributionInLattices:
                for neighborLattice in self.model['neighborProbability'][currentLattice]: 
                    if self.model['neighborProbability'][currentLattice][neighborLattice] > 0: latticeScores[neighborLattice]+=math.log(self.model['hashtagObservingProbability'][currentLattice])+math.log(self.model['neighborProbability'][currentLattice][neighborLattice])
                extraTargetLattices = sorted(latticeScores.iteritems(), key=itemgetter(1))
                while len(targetLattices)<self.params['budget'] and extraTargetLattices:
                    t = extraTargetLattices.pop()
                    if t[0] not in targetLattices:
    #                    print targetLattices+[t[0]]
                        if localityClassId==0 and getRadius([getLocationFromLid(i.replace('_', ' ')) for i in targetLattices+[t[0]]])<=HashtagsClassifier.RADIUS_LIMIT_FOR_LOCAL_HASHTAG_IN_MILES: targetLattices.append(t[0])
        assert len(targetLattices)<=self.params['budget']
        return targetLattices
    
class LocalityClassifier:
    FEATURES_RADIUS = 'radius'
    FEATURES_OCCURANCES_RADIUS = 'occurances_radius'
    FEATURES_AGGGREGATED_OCCURANCES_RADIUS = 'aggregate_occurances_radius'
    classifiersPerformanceFile = hashtagsAnalysisFolder+'/classifiers/classifier_performance'
    def __init__(self, numberOfTimeUnits, features):
        self.clf = None
        self.numberOfTimeUnits = numberOfTimeUnits
        self.features = features
        self.classfierFile = hashtagsClassifiersFolder%(self.features, numberOfTimeUnits)+'model.pkl'
    def build(self, documents):
        X, y = zip(*documents)
        self.clf = SVC(probability=True)
        self.clf.fit(X, y)
        GeneralMethods.runCommand('rm -rf %s*'%self.classfierFile)
        FileIO.createDirectoryForFile(self.classfierFile)
        joblib.dump(self.clf, self.classfierFile)
    def score(self, documents):
        testX, testy = zip(*documents)
        self.clf = self.load()
        return self.clf.score(testX, testy)
    def load(self): 
        if self.clf==None: self.clf = joblib.load(self.classfierFile)
        return self.clf
    def predict(self, ov):
        if self.clf==None: self.clf = joblib.load(self.classfierFile)
        return self.clf.predict(self._getDocument(ov)[0])
    def buildClassifier(self):
        documents = self._getDocuments()
        trainDocuments = documents[:int(len(documents)*0.80)]
        self.build(trainDocuments)
    def _getDocument(self, ov):
        if self.features == LocalityClassifier.FEATURES_RADIUS: return ov.getVector(self.numberOfTimeUnits, radiusOnly=True)
        elif self.features == LocalityClassifier.FEATURES_OCCURANCES_RADIUS: return ov.getVector(self.numberOfTimeUnits, radiusOnly=False)
        elif self.features == LocalityClassifier.FEATURES_AGGGREGATED_OCCURANCES_RADIUS: 
            vector = ov.getVector(self.numberOfTimeUnits, radiusOnly=True, aggregate=True)
            if vector[0][-1]>=HashtagsClassifier.RADIUS_LIMIT_FOR_LOCAL_HASHTAG_IN_MILES: vector[0] = [1]
            else: vector[0] = [0]
            return vector
    def _getDocuments(self):
        documents = []
        for i, h in enumerate(FileIO.iterateJsonFromFile(hashtagsFile%('training_world','%s_%s'%(2,11)))):
            ov = Hashtag(h, dataStructuresToBuildClassifier=True)
            if ov.isValidObject() and ov.classifiable: documents.append(self._getDocument(ov))
#                if self.features == LocalityClassifier.FEATURES_RADIUS: documents.append(ov.getVector(self.numberOfTimeUnits, radiusOnly=True))
#                elif self.features == LocalityClassifier.FEATURES_OCCURANCES_RADIUS: documents.append(ov.getVector(self.numberOfTimeUnits, radiusOnly=False))
#                elif self.features == LocalityClassifier.FEATURES_AGGGREGATED_OCCURANCES_RADIUS: 
#                    vector = ov.getVector(self.numberOfTimeUnits, radiusOnly=True, aggregate=True)
#                    if vector[0][-1]>=HashtagsClassifier.RADIUS_LIMIT_FOR_LOCAL_HASHTAG_IN_MILES: vector[0] = [1]
#                    else: vector[0] = [0]
#                    documents.append(vector)
        return documents
    def testClassifierPerformance(self):
        documents = self._getDocuments()
        testDocuments = documents[-int(len(documents)*0.20):]
        print {'features': self.features, 'numberOfTimeUnits': self.numberOfTimeUnits, 'score': self.score(testDocuments)}
        FileIO.writeToFileAsJson({'features': self.features, 'numberOfTimeUnits': self.numberOfTimeUnits, 'score': self.score(testDocuments)}, LocalityClassifier.classifiersPerformanceFile)
    @staticmethod
    def testClassifierPerformances():
#        GeneralMethods.runCommand('rm -rf %s'%Classifier.classifiersPerformanceFile)
        for numberOfTimeUnits in range(1,25):
            for feature in [LocalityClassifier.FEATURES_AGGGREGATED_OCCURANCES_RADIUS, LocalityClassifier.FEATURES_OCCURANCES_RADIUS, LocalityClassifier.FEATURES_RADIUS]:
                classifier = LocalityClassifier(numberOfTimeUnits, features=feature)
                classifier.testClassifierPerformance()
    @staticmethod
    def buildClassifiers():
        for feature in [LocalityClassifier.FEATURES_AGGGREGATED_OCCURANCES_RADIUS]:
            for numberOfTimeUnits in range(1,25):
                classifier = LocalityClassifier(numberOfTimeUnits, features=feature)
                classifier.buildClassifier()
        
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
                self.occuranceLatticesAggregatedVector = []
                tempOccurances = []
                for t in self.occurances:
                    tempOccurances+=t
                    if t: self.occuranceLatticesVector.append(getRadius(zip(*t)[0]))
                    else: self.occuranceLatticesVector.append(0.0)
                    self.occuranceLatticesAggregatedVector.append(zip(*tempOccurances)[0])
            else: self.classifiable=False
    def getVector(self, length, radiusOnly=True, aggregate=False):
        if len(self.occuranceCountVector)<length: 
            difference = length-len(self.occuranceCountVector)
            self.occuranceCountVector=self.occuranceCountVector+[0 for i in range(difference)]
            self.occuranceLatticesVector=self.occuranceLatticesVector+[0 for i in range(difference)]
            self.occuranceLatticesAggregatedVector=self.occuranceLatticesAggregatedVector+[self.occuranceLatticesAggregatedVector[-1] for i in range(difference)]
        if radiusOnly: vector = self.occuranceLatticesVector[:length]
        else: vector = self.occuranceLatticesVector[:length] + normalize(self.occuranceCountVector[:length])
        if aggregate: vector+=[getRadius(self.occuranceLatticesAggregatedVector[length-1])]
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
        params = dict(budget=5, timeUnitToPickTargetLattices=1)
        params['dataStructuresToBuildClassifier'] = True
#        LatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingTimeUnitToPickTargetLattices()
#        LatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingBudget()
#        LatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateByVaringBudgetAndTimeUnits()
#        GreedyLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingTimeUnitToPickTargetLattices()
#        GreedyLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingBudget()
#        GreedyLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateByVaringBudgetAndTimeUnits()
#        SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingTimeUnitToPickTargetLattices()
#        SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingBudget()
#        SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateByVaringBudgetAndTimeUnits()
#        TransmittingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingTimeUnitToPickTargetLattices()
#        TransmittingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingBudget()
#        SharingProbabilityLatticeSelectionWithLocalityClassifierModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingTimeUnitToPickTargetLattices()
#        SharingProbabilityLatticeSelectionWithLocalityClassifierModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingBudget()
        LatticeSelectionModel.plotModelWithVaryingBudget([LatticeSelectionModel, SharingProbabilityLatticeSelectionModel, SharingProbabilityLatticeSelectionWithLocalityClassifierModel,
                                                                                GreedyLatticeSelectionModel, TransmittingProbabilityLatticeSelectionModel], 
                                                                               Metrics.overall_hit_rate, 
                                                                                   params=params)
#        SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).plotVaringBudgetAndTimeUnits()
        
if __name__ == '__main__':
    Simulation.run()
#    SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), params={})
#    model.saveModelSimulation()
#    LocalityClassifier.testClassifierPerformances()