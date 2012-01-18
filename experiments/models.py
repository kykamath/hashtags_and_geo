'''
Created on Dec 8, 2011

@author: kykamath
'''
import sys
sys.path.append('../')
from library.file_io import FileIO
from settings import hashtagsWithoutEndingWindowFile, hashtagsLatticeGraphFile,\
    hashtagsFile, hashtagsModelsFolder, hashtagsAnalysisFolder,\
    hashtagsClassifiersFolder, us_boundary, sub_world_boundary,\
    hashtagsImagesHastagSharingVsTransmittingProbabilityFolder,\
    targetSelectionRegressionClassifiersFolder
from experiments.mr_area_analysis import getOccuranesInHighestActiveRegion,\
    TIME_UNIT_IN_SECONDS, LATTICE_ACCURACY, HashtagsClassifier,\
    getOccurranceDistributionInEpochs,\
    getRadius
import numpy as np
from library.stats import getOutliersRangeUsingIRQ
from library.geo import getHaversineDistanceForLids, getLatticeLid, getLocationFromLid,\
    plotPointsOnUSMap, plotPointsOnWorldMap, isWithinBoundingBox, getLattice
from collections import defaultdict
from operator import itemgetter
import networkx as nx
from library.graphs import plot, clusterUsingAffinityPropagation
import datetime, math, random
from library.classes import GeneralMethods
import matplotlib.pyplot as plt
from library.plotting import plot3D
from sklearn.svm import SVC
from sklearn.externals import joblib
from itertools import groupby
import matplotlib
from sklearn import linear_model, svm

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
BEST_RATE = 'best_rate'
SHARING_PROBABILITY_LATTICE_SELECTION_MODEL = 'sharing_probability'
TRANSMITTING_PROBABILITY_LATTICE_SELECTION_MODEL = 'transmitting_probability'
SHARING_PROBABILITY_LATTICE_SELECTION_WITH_LOCALITY_CLASSIFIER_MODEL = 'sharing_probability_with_locality_classifier'
LINEAR_REGRESSION_LATTICE_SELECTION_MODEL = 'linear_regression'
SVM_LINEAR_REGRESSION_LATTICE_SELECTION_MODEL = 'svm_linear_regression'
SVM_POLY_REGRESSION_LATTICE_SELECTION_MODEL = 'svm_poly_regression'
SVM_RBF_REGRESSION_LATTICE_SELECTION_MODEL = 'svm_rbf_regression'

class Metrics:
    overall_hit_rate = 'overall_hit_rate'
    hit_rate_after_target_selection = 'hit_rate_after_target_selection'
    miss_rate_before_target_selection = 'miss_rate_before_target_selection'
    best_rate = 'best_rate'
    target_selection_accuracy = 'target_selection_accuracy'
    @staticmethod
    def bestRate(hashtag, **params):
        totalOccurances, occuranceCountInLatices = 0., {}
        if hashtag.occuranceDistributionInTargetLattices:
            selectedTimeUnit = hashtag.occuranceDistributionInTargetLattices.values()[0]['selectedTimeUnit']
            for k,v in hashtag.occuranceDistributionInLattices.iteritems(): 
                totalOccurances+=len(v)
                occuranceCountInLatices[k] = len(filter(lambda i: i>selectedTimeUnit, v))
            return sum(sorted(occuranceCountInLatices.values())[-params['budget']:])/totalOccurances
    @staticmethod
    def targetSelectionAccuracy(hashtag, **params):
        occuranceCountInLatices = {}
        if hashtag.occuranceDistributionInTargetLattices:
            selectedTimeUnit = hashtag.occuranceDistributionInTargetLattices.values()[0]['selectedTimeUnit']
            for k,v in hashtag.occuranceDistributionInLattices.iteritems(): occuranceCountInLatices[k] = len(filter(lambda i: i>selectedTimeUnit, v))
            bestLattices = set(zip(*sorted(occuranceCountInLatices.iteritems(), key=itemgetter(1))[-params['budget']:])[0])
            targetLattices = set(hashtag.occuranceDistributionInTargetLattices.keys())
            return len(bestLattices.intersection(targetLattices))/float(len(bestLattices))
    @staticmethod
    def overallOccurancesHitRate(hashtag, **params):
        totalOccurances, occurancesObserved = 0., 0.
        if hashtag.occuranceDistributionInTargetLattices:
            for k,v in hashtag.occuranceDistributionInLattices.iteritems(): totalOccurances+=len(v)
            for k, v in hashtag.occuranceDistributionInTargetLattices.iteritems(): occurancesObserved+=sum(v['occurances'].values())
            return occurancesObserved/totalOccurances
    @staticmethod
    def occurancesHitRateAfterTargetSelection(hashtag, **params):
        totalOccurances, occurancesObserved = 0., 0.
        if hashtag.occuranceDistributionInTargetLattices:
            targetSelectionTimeUnit = min(v['selectedTimeUnit'] for v in hashtag.occuranceDistributionInTargetLattices.values())
            for k,v in hashtag.occuranceDistributionInLattices.iteritems(): totalOccurances+=len([i for i in v if i>targetSelectionTimeUnit])
            for k, v in hashtag.occuranceDistributionInTargetLattices.iteritems(): occurancesObserved+=sum(v['occurances'].values())
            if totalOccurances!=0.: return occurancesObserved/totalOccurances
            return None
    @staticmethod
    def occurancesMissRateBeforeTargetSelection(hashtag, **params):
        totalOccurances, occurancesBeforeTimeUnit = 0., 0.
        if hashtag.occuranceDistributionInTargetLattices:
            targetSelectionTimeUnit = min(v['selectedTimeUnit'] for v in hashtag.occuranceDistributionInTargetLattices.values())
            for k,v in hashtag.occuranceDistributionInLattices.iteritems(): totalOccurances+=len(v); occurancesBeforeTimeUnit+=len([i for i in v if i<=targetSelectionTimeUnit])
            return occurancesBeforeTimeUnit/totalOccurances
EvaluationMetrics = {
                     Metrics.overall_hit_rate: Metrics.overallOccurancesHitRate,
                     Metrics.hit_rate_after_target_selection: Metrics.occurancesHitRateAfterTargetSelection,
                     Metrics.miss_rate_before_target_selection: Metrics.occurancesMissRateBeforeTargetSelection,
#                     Metrics.best_rate: Metrics.bestRate,
                      Metrics.target_selection_accuracy: Metrics.targetSelectionAccuracy  
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
                hashtags[hashtag.hashtagObject['h']] = {'model': self.id, 'classId': hashtag.hashtagClassId, 'metrics': dict([(k, method(hashtag, budget=self.budget))for k,method in EvaluationMetrics.iteritems()])}
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
                plt.plot(dataX, dataY, label=model.id, lw=2)
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
                plt.plot(dataX, dataY, label=model.id, lw=2)
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

class BestRateModel(LatticeSelectionModel):
    ''' Pick the location with maximum observations till that time.
    '''
    def __init__(self, **kwargs): super(BestRateModel, self).__init__(BEST_RATE, **kwargs)


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
    @staticmethod
    def plotClassifierPerformance():
        featuresData = defaultdict(dict)
        for data in FileIO.iterateJsonFromFile(LocalityClassifier.classifiersPerformanceFile):
            featuresData[data['features']][data['numberOfTimeUnits']] = data['score']
        for id, v in featuresData.iteritems():
            dataX, dataY = sorted(v.keys()), [v[k] for k in sorted(v.keys())]
            plt.plot(dataX, dataY, lw=2, label=id)
        plt.legend(loc=3)
        plt.ylim(ymin=0.4)
        plt.title('Locality classifier performance')
        plt.show()
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
def plotLocationClustersOnMap(title, graph):
    noOfClusters, clusters = clusterUsingAffinityPropagation(graph)
    nodeToClusterIdMap = dict(clusters)
    colorMap = dict([(i, GeneralMethods.getRandomColor()) for i in range(noOfClusters)])
    clusters = [(c, list(l)) for c, l in groupby(sorted(clusters, key=itemgetter(1)), key=itemgetter(1))]
    points, colors = zip(*map(lambda  l: (getLocationFromLid(l.replace('_', ' ')), colorMap[nodeToClusterIdMap[l]]), graph.nodes()))
    _, m =plotPointsOnWorldMap(points[:1], s=0, lw=0, c=colors[:1], returnBaseMapObject=True)
    for u, v, data in graph.edges(data=True):
        if nodeToClusterIdMap[u]==nodeToClusterIdMap[v]:
            color, u, v, w = colorMap[nodeToClusterIdMap[u]], getLocationFromLid(u.replace('_', ' ')), getLocationFromLid(v.replace('_', ' ')), data['w']
            m.drawgreatcircle(u[1],u[0],v[1],v[0],color=color, alpha=0.5)
    plt.title(title)
    plt.show()
def plotLocationGraphOnMap(title, graph):
#    noOfClusters, clusters = clusterUsingAffinityPropagation(graph)
#    nodeToClusterIdMap = dict(clusters)
#    colorMap = dict([(i, GeneralMethods.getRandomColor()) for i in range(noOfClusters)])
#    clusters = [(c, list(l)) for c, l in groupby(sorted(clusters, key=itemgetter(1)), key=itemgetter(1))]
#    points, colors = zip(*map(lambda  l: (getLocationFromLid(l.replace('_', ' ')), 'k'), graph.nodes()))
#    _, m =plotPointsOnWorldMap(points[:1], s=0, lw=0, c=colors[:1], returnBaseMapObject=True)
#    for u, v, data in graph.edges(data=True):
#        if nodeToClusterIdMap[u]==nodeToClusterIdMap[v]:
#            color, u, v, w = colorMap[nodeToClusterIdMap[u]], getLocationFromLid(u.replace('_', ' ')), getLocationFromLid(v.replace('_', ' ')), data['w']
#            m.drawgreatcircle(u[1],u[0],v[1],v[0],color='k', alpha=0.5)
#    plt.title(title)
#    plt.show()
    
    noOfClusters, clusters = clusterUsingAffinityPropagation(graph)
    nodeToClusterIdMap = dict(clusters)
    colorMap = dict([(i, GeneralMethods.getRandomColor()) for i in range(noOfClusters)])
    clusters = [(c, list(l)) for c, l in groupby(sorted(clusters, key=itemgetter(1)), key=itemgetter(1))]
    points, colors = zip(*map(lambda  l: (getLocationFromLid(l.replace('_', ' ')), colorMap[nodeToClusterIdMap[l]]), graph.nodes()))
    _, m =plotPointsOnWorldMap(points[:1], s=0, lw=0, c=colors[:1], returnBaseMapObject=True)
    for u, v, data in graph.edges(data=True):
        if nodeToClusterIdMap[u]==nodeToClusterIdMap[v]:
            color, u, v, w = colorMap[nodeToClusterIdMap[u]], getLocationFromLid(u.replace('_', ' ')), getLocationFromLid(v.replace('_', ' ')), data['w']
            m.drawgreatcircle(u[1],u[0],v[1],v[0],color='k', alpha=0.5)
    plt.title(title)
    plt.show()
    
class Analysis:
    @staticmethod
    def analyzeLatticeProbabilityGraph():
        params = dict(budget=5, timeUnitToPickTargetLattices=1)
        model = SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params)
#        model = TransmittingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params)
        graph = nx.DiGraph()
        for currentLattice in model.model['neighborProbability']:
            for neighborLattice in model.model['neighborProbability'][currentLattice]: 
                if isWithinBoundingBox(getLocationFromLid(currentLattice.replace('_', ' ')), sub_world_boundary) and \
                    isWithinBoundingBox(getLocationFromLid(neighborLattice.replace('_', ' ')), sub_world_boundary):
                    graph.add_edge(currentLattice, neighborLattice, 
                                   {'w':model.model['hashtagObservingProbability'][currentLattice]*model.model['neighborProbability'][currentLattice][neighborLattice]})
#        nodesToRemove = sorted([(c, model.model['hashtagObservingProbability'][c]) for c in model.model['hashtagObservingProbability'] if isWithinBoundingBox(getLocationFromLid(c.replace('_', ' ')), sub_world_boundary)], 
#                               key=itemgetter(1))[:int(0.75*graph.number_of_nodes())]
#        for u, _ in nodesToRemove: graph.remove_node(u)
        
        edgesToRemove = sorted(graph.edges_iter(data=True),key=lambda t:t[2]['w'])[:int(0.75*graph.number_of_edges())]
        for u,v,_ in edgesToRemove: graph.remove_edge(u, v)
        for u in graph.nodes()[:]:
            if graph.degree(u)==0: graph.remove_node(u)
        plotLocationClustersOnMap(model.id, graph)
#        plotLocationGraphOnMap(model.id, graph)

    @staticmethod
    def plotSharingAndTransmittingProbabilityForLatticesOnMap():
        def plotOnMap(lattice, points, colors):
            cm = matplotlib.cm.get_cmap('YlOrRd')
            sc = plotPointsOnWorldMap(points, c=colors, cmap=cm, lw = 0, vmin=0.0)
            plotPointsOnWorldMap([getLocationFromLid(lattice.replace('_', ' '))], c='#00FF00', lw = 0)
            plt.colorbar(sc)
        params = dict(budget=5, timeUnitToPickTargetLattices=1)
        transmittingModel = TransmittingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params)
        sharingModel = SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params)
        FileIO.createDirectoryForFile(hashtagsImagesHastagSharingVsTransmittingProbabilityFolder%'world')
        for count, lattice in enumerate(transmittingModel.model['neighborProbability']):
            transmittingPoints, transmittingColors = zip(*sorted([(getLocationFromLid(neighborId.replace('_', ' ')), val) for neighborId, val in transmittingModel.model['neighborProbability'][lattice].iteritems()], key=itemgetter(1)))
            sharingPoints, sharingColors = zip(*sorted([(getLocationFromLid(neighborId.replace('_', ' ')), val) for neighborId, val in sharingModel.model['neighborProbability'][lattice].iteritems()], key=itemgetter(1)))
            transmittingColors = [float(i)/len(transmittingColors) for i in range(1, len(transmittingColors)+1)]
            sharingColors = [float(i)/len(sharingColors) for i in range(1, len(sharingColors)+1)]
            plt.subplot(211), plotOnMap(lattice, transmittingPoints, transmittingColors), plt.xlabel(transmittingModel.id), plt.title(lattice)
            plt.subplot(212), plotOnMap(lattice, sharingPoints, sharingColors), plt.xlabel(sharingModel.id)
            print count, hashtagsImagesHastagSharingVsTransmittingProbabilityFolder%'world'+'%s.png'%lattice
            plt.savefig(hashtagsImagesHastagSharingVsTransmittingProbabilityFolder%'world'+'%s.png'%lattice)
#            plt.show()
            plt.clf()
            
    @staticmethod
    def run():
        Analysis.analyzeLatticeProbabilityGraph()
#        Analysis.plotSharingAndTransmittingProbabilityForLatticesOnMap()

class TargetSelectionRegressionClassifier(object):
    classifiersPerformanceFile = hashtagsAnalysisFolder+'/ts_classifiers/classifier_performance'
    def __init__(self, id='linear_regression', decisionTimeUnit=None, predictingLattice=None): 
        self.id = id
        self.decisionTimeUnit = decisionTimeUnit
        self.predictingLattice = predictingLattice
        self.classfierFile = targetSelectionRegressionClassifiersFolder%(self.id, self.decisionTimeUnit, self.predictingLattice)+'model.pkl'
        FileIO.createDirectoryForFile(self.classfierFile)
        self.clf = None
    def buildClassifier(self, trainingDocuments):
        inputVectors, outputValues = zip(*trainingDocuments)
        self.clf = linear_model.LinearRegression()
        self.clf.fit(inputVectors, outputValues)
    def build(self, trainingDocuments):
        self.buildClassifier(trainingDocuments)
        GeneralMethods.runCommand('rm -rf %s*'%self.classfierFile)
        FileIO.createDirectoryForFile(self.classfierFile)
        joblib.dump(self.clf, self.classfierFile)
    def predict(self, vector):
        if self.clf==None: self.clf = joblib.load(self.classfierFile)
        return self.clf.predict(vector)
    @staticmethod
    def writeLattices():
        validLattices = set()
        for data in FileIO.iterateJsonFromFile(hashtagsLatticeGraphFile%('world','%s_%s'%(2,11))): validLattices.add(data['id'])
        lattices = set()
        for h in FileIO.iterateJsonFromFile(hashtagsFile%('training_world','%s_%s'%(2,11))): 
            hashtag = Hashtag(h)
            if hashtag.isValidObject():
                for timeUnit, occs in enumerate(hashtag.getOccrancesEveryTimeWindowIterator(HashtagsClassifier.CLASSIFIER_TIME_UNIT_IN_SECONDS)):
                    occs = filter(lambda t: t[0] in validLattices, occs)
                    occs = sorted(occs, key=itemgetter(0))
                    if occs: 
                        for lattice in zip(*occs)[0]: lattices.add(lattice)
        lattices = sorted(list(lattices))
        FileIO.writeToFileAsJson(lattices, '../data/lattices.json')
    @staticmethod
    def loadLattices(): return list(FileIO.iterateJsonFromFile('../data/lattices.json'))[0]
    @staticmethod
    def getPercentageDistributionInLattice(document):
        data = zip(*document)[1]
        distributionInLaticces = defaultdict(int)
        for d in data:
            for k, v in d: distributionInLaticces[k]+=v
        total = float(sum(distributionInLaticces.values()))
        return dict([k,v/total] for k, v in distributionInLaticces.iteritems())
    
class TargetSelectionRegressionSVMRBFClassifier(TargetSelectionRegressionClassifier):
    def __init__(self, id='svm_rbf_regression', decisionTimeUnit=None, predictingLattice=None):
        TargetSelectionRegressionClassifier.__init__(self, id=id, decisionTimeUnit=decisionTimeUnit, predictingLattice=predictingLattice)
    def buildClassifier(self, trainingDocuments):
        inputVectors, outputValues = zip(*trainingDocuments)
        self.clf = svm.SVR(kernel='rbf', C=1e4, gamma=0.1)
        self.clf.fit(inputVectors, outputValues)
class TargetSelectionRegressionSVMLinearClassifier(TargetSelectionRegressionClassifier):
    def __init__(self, id='svm_linear_regression', decisionTimeUnit=None, predictingLattice=None):
        TargetSelectionRegressionClassifier.__init__(self, id=id, decisionTimeUnit=decisionTimeUnit, predictingLattice=predictingLattice)
    def buildClassifier(self, trainingDocuments):
        inputVectors, outputValues = zip(*trainingDocuments)
        self.clf = svm.SVR(kernel='linear', C=1e4)
        self.clf.fit(inputVectors, outputValues)
class TargetSelectionRegressionSVMPolyClassifier(TargetSelectionRegressionClassifier):
    def __init__(self, id='svm_poly_regression', decisionTimeUnit=None, predictingLattice=None):
        TargetSelectionRegressionClassifier.__init__(self, id=id, decisionTimeUnit=decisionTimeUnit, predictingLattice=predictingLattice)
    def buildClassifier(self, trainingDocuments):
        inputVectors, outputValues = zip(*trainingDocuments)
        self.clf = svm.SVR(kernel='poly', C=1e4, degree=2)
        self.clf.fit(inputVectors, outputValues)
class LinearRegressionLatticeSelectionModel(LatticeSelectionModel):
    ''' Pick the location using linear regression.
    '''
    lattices = TargetSelectionRegressionClassifier.loadLattices()
    def __init__(self, **kwargs): 
        super(LinearRegressionLatticeSelectionModel, self).__init__(LINEAR_REGRESSION_LATTICE_SELECTION_MODEL, **kwargs)
        self.regressionClassType = TargetSelectionRegressionClassifier
    def selectTargetLattices(self, currentTimeUnit, hashtag): 
        occuranceDistributionInLattices = dict([(k, len(v)) for k, v in hashtag.occuranceDistributionInLattices.iteritems()])
        total = float(sum(occuranceDistributionInLattices.values()))
        occuranceDistributionInLattices = dict([k,v/total] for k, v in occuranceDistributionInLattices.iteritems())
        vector =  [occuranceDistributionInLattices.get(l, 0) for l in LinearRegressionLatticeSelectionModel.lattices]
        latticeScores = [(l, self.regressionClassType(decisionTimeUnit=currentTimeUnit+1, predictingLattice=l).predict(vector)) for l in LinearRegressionLatticeSelectionModel.lattices]
        return zip(*sorted(latticeScores, key=itemgetter(1), reverse=True)[:self.params['budget']])[0]
    
class SVMLinearRegressionLatticeSelectionModel(LinearRegressionLatticeSelectionModel):
    def __init__(self, **kwargs): 
        super(SVMLinearRegressionLatticeSelectionModel, self).__init__(SVM_LINEAR_REGRESSION_LATTICE_SELECTION_MODEL, **kwargs)
        self.regressionClassType = TargetSelectionRegressionSVMLinearClassifier

class Simulation:
    trainingHashtagsFile = hashtagsFile%('training_world','%s_%s'%(2,11))
    testingHashtagsFile = hashtagsFile%('testing_world','%s_%s'%(2,11))
    @staticmethod
    def run():
        params = dict(budget=5, timeUnitToPickTargetLattices=1)
        params['dataStructuresToBuildClassifier'] = True
#        BestRateModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingTimeUnitToPickTargetLattices()
#        BestRateModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingBudget()
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

#        LinearRegressionLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingTimeUnitToPickTargetLattices()
#        LinearRegressionLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingBudget()
        SVMLinearRegressionLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingTimeUnitToPickTargetLattices()
#        SVMLinearRegressionLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).evaluateModelWithVaryingBudget()

#        LatticeSelectionModel.plotModelWithVaryingTimeUnitToPickTargetLattices([LatticeSelectionModel, SharingProbabilityLatticeSelectionModel, SharingProbabilityLatticeSelectionWithLocalityClassifierModel,
#                                                                                GreedyLatticeSelectionModel, TransmittingProbabilityLatticeSelectionModel], 
#                                                                               Metrics.target_selection_accuracy, 
#                                                                                   params=params)
#        SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), testingHashtagsFile=Simulation.testingHashtagsFile, params=params).plotVaringBudgetAndTimeUnits()
        
if __name__ == '__main__':
    Simulation.run()
#    Analysis.run()
#    LocalityClassifier.plotClassifierPerformance()
#    SharingProbabilityLatticeSelectionModel(folderType='training_world', timeRange=(2,11), params={})
#    model.saveModelSimulation()
#    LocalityClassifier.testClassifierPerformances()