'''
Created on Feb 14, 2012

@author: kykamath

@todo: For the probabilties make sure we are not limited by numberOfHashtags.

'''
from analysis import iterateJsonFromFile
from settings import timeUnitWithOccurrencesFile, locationsGraphFile,\
                     modelsFolder, hashtagsAllOccurrencesWithinWindowFile
from datetime import datetime, timedelta
from mr_analysis import TIME_UNIT_IN_SECONDS
import time, random, math, inspect
from collections import defaultdict
from itertools import groupby, chain
from operator import itemgetter
from library.classes import GeneralMethods, timeit
from library.file_io import FileIO
import numpy as np
from library.stats import getOutliersRangeUsingIRQ
import matplotlib.pyplot as plt
from multiprocessing import Pool
from library.geo import getLocationFromLid, getHaversineDistance

NAN_VALUE = -1.0

LOCATIONS_LIST, SHARING_PROBABILITIES, TRANSMITTING_PROBABILITIES = None, None, None

def filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeHashtags, neighborHashtags, findLag=True):
    if findLag: 
        dataToReturn = [(hashtag, np.abs(latticeHashtags[hashtag][0]-timeTuple[0])/TIME_UNIT_IN_SECONDS) for hashtag, timeTuple in neighborHashtags.iteritems() if hashtag in latticeHashtags]
        _, upperRangeForTemporalDistance = getOutliersRangeUsingIRQ(zip(*(dataToReturn))[1])
        return dict(filter(lambda t: t[1]<=upperRangeForTemporalDistance, dataToReturn))
    else: 
        dataToReturn = [(hashtag, timeTuple, np.abs(latticeHashtags[hashtag][0]-timeTuple[0])/TIME_UNIT_IN_SECONDS) for hashtag, timeTuple in neighborHashtags.iteritems() if hashtag in latticeHashtags]
        _, upperRangeForTemporalDistance = getOutliersRangeUsingIRQ(zip(*(dataToReturn))[2])
        return dict([(t[0], t[1]) for t in dataToReturn if t[2]<=upperRangeForTemporalDistance])

def loadLocationsList():
    global LOCATIONS_LIST
    if not LOCATIONS_LIST: LOCATIONS_LIST = [latticeObject['id'] for latticeObject in FileIO.iterateJsonFromFile(locationsGraphFile)]

def loadSharingProbabilities():
    global SHARING_PROBABILITIES
    if not SHARING_PROBABILITIES:
        SHARING_PROBABILITIES = {'neighborProbability': defaultdict(dict), 'hashtagObservingProbability': {}}
        hashtagsObserved = []
        for latticeObject in FileIO.iterateJsonFromFile(locationsGraphFile):
            latticeHashtagsSet = set(latticeObject['hashtags'])
            hashtagsObserved+=latticeObject['hashtags']
            SHARING_PROBABILITIES['hashtagObservingProbability'][latticeObject['id']] = latticeHashtagsSet
            for neighborLattice, neighborHashtags in latticeObject['links'].iteritems():
                neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags)
                neighborHashtagsSet = set(neighborHashtags)
                SHARING_PROBABILITIES['neighborProbability'][latticeObject['id']][neighborLattice]=len(latticeHashtagsSet.intersection(neighborHashtagsSet))/float(len(latticeHashtagsSet))
            SHARING_PROBABILITIES['neighborProbability'][latticeObject['id']][latticeObject['id']]=1.0
        totalNumberOfHashtagsObserved=float(len(set(hashtagsObserved)))
        for lattice in SHARING_PROBABILITIES['hashtagObservingProbability'].keys()[:]: SHARING_PROBABILITIES['hashtagObservingProbability'][lattice] = len(SHARING_PROBABILITIES['hashtagObservingProbability'][lattice])/totalNumberOfHashtagsObserved
    return SHARING_PROBABILITIES

def loadTransmittingProbabilities():
    global TRANSMITTING_PROBABILITIES
    if not TRANSMITTING_PROBABILITIES:
        TRANSMITTING_PROBABILITIES = {'neighborProbability': defaultdict(dict), 'hashtagObservingProbability': {}}
        hashtagsObserved = []
        for latticeObject in FileIO.iterateJsonFromFile(locationsGraphFile):
            latticeHashtagsSet = set(latticeObject['hashtags'])
            hashtagsObserved+=latticeObject['hashtags']
            TRANSMITTING_PROBABILITIES['hashtagObservingProbability'][latticeObject['id']] = latticeHashtagsSet
            for neighborLattice, neighborHashtags in latticeObject['links'].iteritems():
                neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags, findLag=False)
    #                neighborHashtagsSet = set(neighborHashtags)
                transmittedHashtags = [k for k in neighborHashtags if k in latticeObject['hashtags'] and latticeObject['hashtags'][k][0]<neighborHashtags[k][0]]
                TRANSMITTING_PROBABILITIES['neighborProbability'][latticeObject['id']][neighborLattice]=len(transmittedHashtags)/float(len(latticeHashtagsSet))
            TRANSMITTING_PROBABILITIES['neighborProbability'][latticeObject['id']][latticeObject['id']]=1.0
        totalNumberOfHashtagsObserved=float(len(set(hashtagsObserved)))
        for lattice in TRANSMITTING_PROBABILITIES['hashtagObservingProbability'].keys()[:]: TRANSMITTING_PROBABILITIES['hashtagObservingProbability'][lattice] = len(TRANSMITTING_PROBABILITIES['hashtagObservingProbability'][lattice])/totalNumberOfHashtagsObserved
    return TRANSMITTING_PROBABILITIES

class CoverageModel():
    @staticmethod
    def _probabilityDistributionForLattices(points):
            points = sorted(points, key=itemgetter(0,1))
            numberOfOccurrences = float(len(points))
            return [(k, len(list(data))/numberOfOccurrences) for k, data in groupby(points, key=itemgetter(0,1))]
    @staticmethod
    def _spreadingFunction(currentLattice, sourceLattice, probabilityAtSourceLattice): return 1.01**(-getHaversineDistance(currentLattice, sourceLattice))*probabilityAtSourceLattice
    @staticmethod
    def spreadProbability(points):
        latticeScores = {}
        probabilityDistributionForObservedLattices = CoverageModel._probabilityDistributionForLattices(points)
        for lattice in LOCATIONS_LIST:
            score = 0.0
            currentLattice = getLocationFromLid(lattice.replace('_', ' '))
            latticeScores[lattice] = sum([CoverageModel._spreadingFunction(currentLattice, sourceLattice, probabilityAtSourceLattice)for sourceLattice, probabilityAtSourceLattice in probabilityDistributionForObservedLattices])
        total = sum(latticeScores.values())
        for k in latticeScores: 
            if total==0: latticeScores[k]=1.0
            else: latticeScores[k]/=total
        return latticeScores
    @staticmethod
    def spreadDistance(points):
        latticeScores = {}
        distribution_in_observed_lattices = [(k, len(list(data))) for k, data in groupby(sorted(points, key=itemgetter(0,1)), key=itemgetter(0,1))]
        for lattice in LOCATIONS_LIST:
            currentLattice = getLocationFromLid(lattice.replace('_', ' '))
            latticeScores[lattice] = sum([CoverageModel._spreadingFunction(currentLattice, sourceLattice, count_at_source_lattice)for sourceLattice, count_at_source_lattice in distribution_in_observed_lattices])
        return latticeScores
        
class Propagations:
    def __init__(self, startTime, interval):
        self.startTime, self.interval = startTime, interval
        self.occurrences = defaultdict(list)
    def update(self, occurrences):
        for h, loc, t in occurrences: self.occurrences[loc].append([h, t])
    def getCoverageProbabilities(self): 
        if self.occurrences:
            occurrences = chain(*[zip(zip(*l)[0], [getLocationFromLid(k.replace('_', ' '))]*len(l)) for k,l in self.occurrences.iteritems()])
            return dict([(k, CoverageModel.spreadProbability(zip(*l)[1])) for k, l in groupby(sorted(occurrences, key=itemgetter(0)), key=itemgetter(0))])
    def getCoverageDistances(self):
        if self.occurrences:
            occurrences = chain(*[zip(zip(*l)[0], [getLocationFromLid(k.replace('_', ' '))]*len(l)) for k,l in self.occurrences.iteritems()])
            return dict([(k, CoverageModel.spreadDistance(zip(*l)[1])) for k, l in groupby(sorted(occurrences, key=itemgetter(0)), key=itemgetter(0))])

class EvaluationMetrics:
    ACCURACY = 'accuracy'
    IMPACT = 'impact'
    IMPACT_DIFFERENCE = 'impact_difference'
    @staticmethod
    def _bestHashtagsForLocation(actualPropagation, **conf):
        bestHashtagsForLocation = {}
        for loc, occs in actualPropagation.occurrences.iteritems():
            bestHashtagsForLocation[loc] = zip(*sorted([(h, len(list(hOccs)))for h, hOccs in groupby(sorted(occs, key=itemgetter(0)), key=itemgetter(0))], key=itemgetter(1)))[0][-conf['noOfTargetHashtags']:]
        return bestHashtagsForLocation
    @staticmethod
    def _impact(loc, hashtags, actualPropagation):
        if loc in actualPropagation.occurrences: 
            totalOccs = len(actualPropagation.occurrences[loc])
            occsOfTargetHashtags = len([h for h, t in actualPropagation.occurrences[loc] if h in hashtags])
            return float(occsOfTargetHashtags)/totalOccs
        else: return NAN_VALUE
    @staticmethod
    def accuracy(hashtagsForLocation, actualPropagation, *args, **kwargs):
        bestHashtagsForLocation, metricScorePerLocation = EvaluationMetrics._bestHashtagsForLocation(actualPropagation, **kwargs), {}
        for loc, hashtags in hashtagsForLocation.iteritems(): 
            bestSet = set(bestHashtagsForLocation.get(loc, []))
            if bestSet: metricScorePerLocation[loc] = len(set(hashtags).intersection(bestSet))/float(len(bestSet))
#            else: metricScorePerLocation[loc] = NAN_VALUE
        return metricScorePerLocation
    @staticmethod
    def impact(hashtagsForLattice, actualPropagation, *args, **kwargs):
        metricScorePerLocation = {}
        for loc, hashtags in hashtagsForLattice.iteritems(): 
            score_for_predicted_hashtags = EvaluationMetrics._impact(loc, hashtags, actualPropagation)
            if score_for_predicted_hashtags!=NAN_VALUE: metricScorePerLocation[loc] = score_for_predicted_hashtags
        return metricScorePerLocation
    @staticmethod
    def impactDifference(hashtagsForLattice, actualPropagation, *args, **kwargs):
        bestHashtagsForLocation, metricScorePerLocation = EvaluationMetrics._bestHashtagsForLocation(actualPropagation, **kwargs), {}
        for loc, hashtags in hashtagsForLattice.iteritems(): 
            score_for_best_hashtags = EvaluationMetrics._impact(loc, bestHashtagsForLocation.get(loc, []), actualPropagation)
            if score_for_best_hashtags != NAN_VALUE: 
                score_for_predicted_hashtags = EvaluationMetrics._impact(loc, hashtags, actualPropagation)
                metricScorePerLocation[loc] = score_for_best_hashtags - score_for_predicted_hashtags
        return metricScorePerLocation
EVALUATION_METRIC_METHODS = dict([
                                  (EvaluationMetrics.ACCURACY, EvaluationMetrics.accuracy),
                                  (EvaluationMetrics.IMPACT, EvaluationMetrics.impact),
                                  (EvaluationMetrics.IMPACT_DIFFERENCE, EvaluationMetrics.impactDifference),
                            ])

class PredictionModels:
    RANDOM = 'random'
    GREEDY = 'greedy'
    SHARING_PROBABILITY = 'sharing_probability'
    TRANSMITTING_PROBABILITY = 'transmitting_probability'
    COVERAGE_PROBABILITY = 'coverage_probability'
    SHARING_PROBABILITY_WITH_COVERAGE = 'sharing_probability_with_coverage'
    TRANSMITTING_PROBABILITY_WITH_COVERAGE = 'transmitting_probability_with_coverage'
    COVERAGE_DISTANCE = 'coverage_distance'
    SHARING_PROBABILITY_WITH_COVERAGE_DISTANCE = 'sharing_probability_with_coverage_distance'
    TRANSMITTING_PROBABILITY_WITH_COVERAGE_DISTANCE = 'transmitting_probability_with_coverage_distance'
    @staticmethod
    def _hashtag_distribution_in_locations(occurrences):
        hashtag_distribution, hashtag_distribution_in_locations = defaultdict(dict), defaultdict(dict)
        for location, occs in occurrences.iteritems():
            for h, _ in occs: 
                if location not in hashtag_distribution[h]: hashtag_distribution[h][location] = 0
                hashtag_distribution[h][location]+=1
        for h in hashtag_distribution.keys()[:]: 
#            total_occurrences = float(sum(hashtag_distribution[h].values()))
#            for l, v in hashtag_distribution[h].iteritems(): hashtag_distribution_in_locations[l][h] = v/total_occurrences
            for l, v in hashtag_distribution[h].iteritems(): hashtag_distribution_in_locations[l][h] = v
        return hashtag_distribution_in_locations
    @staticmethod
    def _hashtags_by_location_probabilities(propagation_for_prediction, location_probabilities, *args, **conf):
        hashtags_for_lattice = defaultdict(list)
        hashtag_distribution_in_locations = PredictionModels._hashtag_distribution_in_locations(propagation_for_prediction.occurrences)
        if propagation_for_prediction.occurrences:
            for loc in LOCATIONS_LIST:
                hashtag_scores, hashtags = defaultdict(float), []
                for neighboring_location in location_probabilities['neighborProbability'][loc]:
                    if location_probabilities['neighborProbability'][loc][neighboring_location]!=0.0:
    #                    for h in hashtag_distribution_in_locations[loc]: hashtag_scores[h]+=math.log(hashtag_distribution_in_locations[loc][h]) + math.log(SHARING_PROBABILITIES['neighborProbability'][loc][neighboring_location])
                        for h in hashtag_distribution_in_locations[neighboring_location]: 
#                            hashtag_scores[h]+=math.log(hashtag_distribution_in_locations[neighboring_location][h]) + math.log(location_probabilities['neighborProbability'][loc][neighboring_location])
#                            hashtag_scores[h]+=(hashtag_distribution_in_locations[neighboring_location][h] * location_probabilities['neighborProbability'][loc][neighboring_location])
                            hashtag_scores[h]+=location_probabilities['neighborProbability'][loc][neighboring_location]
                hashtags_for_lattice[loc] = []
#                if loc in propagation_for_prediction.occurrences:
#                    occs = propagation_for_prediction.occurrences[loc]
#                    hashtags_for_lattice[loc] = list(zip(*sorted([(h, len(list(hOccs)))for h, hOccs in groupby(sorted(occs, key=itemgetter(0)), key=itemgetter(0))], key=itemgetter(1)))[0][-conf['noOfTargetHashtags']:])
                if hashtag_scores: hashtags = list(zip(*sorted(hashtag_scores.iteritems(), key=itemgetter(1)))[0])
                while len(hashtags_for_lattice[loc])<conf['noOfTargetHashtags'] and hashtags:
                    h = hashtags.pop()
                    if h not in hashtags_for_lattice[loc]: hashtags_for_lattice[loc].append(h)
        return hashtags_for_lattice
    @staticmethod
    def _hashtags_by_location_and_coverage_probabilities(propagation_for_prediction, location_probabilities, hashtag_coverage_probabilities, *args, **conf):
        hashtags_for_lattice = defaultdict(list)
        hashtag_distribution_in_locations = PredictionModels._hashtag_distribution_in_locations(propagation_for_prediction.occurrences)
        if propagation_for_prediction.occurrences:
            for loc, occs in propagation_for_prediction.occurrences.iteritems():
                hashtag_scores, hashtags = defaultdict(float), []
                for neighboring_location in location_probabilities['neighborProbability'][loc]:
                    if location_probabilities['neighborProbability'][loc][neighboring_location]!=0.0:
                        for h in hashtag_distribution_in_locations[neighboring_location]: 
#                            hashtag_scores[h]+=math.log(hashtag_distribution_in_locations[neighboring_location][h]) + math.log(location_probabilities['neighborProbability'][loc][neighboring_location])
                            hashtag_scores[h]+=(hashtag_coverage_probabilities[h][neighboring_location] * location_probabilities['neighborProbability'][loc][neighboring_location])
                hashtags_for_lattice[loc] = list(zip(*sorted([(h, len(list(hOccs)))for h, hOccs in groupby(sorted(occs, key=itemgetter(0)), key=itemgetter(0))], key=itemgetter(1)))[0][-conf['noOfTargetHashtags']:])
                if hashtag_scores: 
#                    hashtags = list(zip(*sorted(hashtag_scores.iteritems(), key=itemgetter(1)))[0][-conf['noOfTargetHashtags']:])
                    hashtags = list(zip(*sorted(hashtag_scores.iteritems(), key=itemgetter(1)))[0])
                while len(hashtags_for_lattice[loc])<conf['noOfTargetHashtags'] and hashtags:
                    h = hashtags.pop()
                    if h not in hashtags_for_lattice[loc]: hashtags_for_lattice[loc].append(h)
        return hashtags_for_lattice
    @staticmethod
    def random(propagation_for_prediction, *args, **conf):
        hashtags_for_lattice = defaultdict(list)
        if propagation_for_prediction.occurrences:
            for loc, occs in propagation_for_prediction.occurrences.iteritems():
                uniqueHashtags = set(zip(*occs)[0])
                hashtags_for_lattice[loc] = random.sample(uniqueHashtags, min(len(uniqueHashtags), conf['noOfTargetHashtags']))
        return hashtags_for_lattice
    @staticmethod
    def greedy(propagation_for_prediction, *args, **conf):
        hashtags_for_lattice = defaultdict(list)
        if propagation_for_prediction.occurrences:
            for loc, occs in propagation_for_prediction.occurrences.iteritems():
                hashtags_for_lattice[loc] = zip(*sorted([(h, len(list(hOccs)))for h, hOccs in groupby(sorted(occs, key=itemgetter(0)), key=itemgetter(0))], key=itemgetter(1)))[0][-conf['noOfTargetHashtags']:]
        return hashtags_for_lattice
    @staticmethod
    def sharing_probability(propagation_for_prediction, *args, **conf): loadSharingProbabilities(); return PredictionModels._hashtags_by_location_probabilities(propagation_for_prediction, SHARING_PROBABILITIES, *args, **conf)
    @staticmethod
    def transmitting_probability(propagation_for_prediction, *args, **conf): loadTransmittingProbabilities(); return PredictionModels._hashtags_by_location_probabilities(propagation_for_prediction, TRANSMITTING_PROBABILITIES, *args, **conf)
    @staticmethod
    def coverage_probability(propagation_for_prediction, *args, **conf): 
        hashtags_for_lattice = defaultdict(list)
        hashtag_coverage_probabilities = propagation_for_prediction.getCoverageProbabilities()
        if hashtag_coverage_probabilities:
            hashtag_scores_for_location = {}
            for location in LOCATIONS_LIST:
                hashtag_scores_for_location = dict([(hashtag, hashtag_coverage_probabilities[hashtag][location]) for hashtag in hashtag_coverage_probabilities])
                hashtags_for_lattice[location] = zip(*sorted(hashtag_scores_for_location.iteritems(), key=itemgetter(1)))[0][-conf['noOfTargetHashtags']:]
        return hashtags_for_lattice
    @staticmethod
    def sharing_probability_with_coverage(propagation_for_prediction, *args, **conf): 
        loadSharingProbabilities()
        hashtag_coverage_probabilities = propagation_for_prediction.getCoverageProbabilities()
        return PredictionModels._hashtags_by_location_and_coverage_probabilities(propagation_for_prediction, SHARING_PROBABILITIES, hashtag_coverage_probabilities, *args, **conf)
    @staticmethod
    def transmitting_probability_with_coverage(propagation_for_prediction, *args, **conf): 
        loadTransmittingProbabilities()
        hashtag_coverage_probabilities = propagation_for_prediction.getCoverageProbabilities()
        return PredictionModels._hashtags_by_location_and_coverage_probabilities(propagation_for_prediction, TRANSMITTING_PROBABILITIES, hashtag_coverage_probabilities, *args, **conf)
    @staticmethod
    def coverage_distance(propagation_for_prediction, *args, **conf): 
        hashtags_for_lattice = defaultdict(list)
        coverage_distances_for_hashtags = propagation_for_prediction.getCoverageDistances()
        if coverage_distances_for_hashtags:
            for location in LOCATIONS_LIST:
                hashtag_scores_for_location = dict([(hashtag, coverage_distances_for_hashtags[hashtag][location]) for hashtag in coverage_distances_for_hashtags])
                total_score = sum(hashtag_scores_for_location.values())
                for hashtag in hashtag_scores_for_location: hashtag_scores_for_location[hashtag]/=total_score
                hashtags_for_lattice[location] = zip(*sorted(hashtag_scores_for_location.iteritems(), key=itemgetter(1)))[0][-conf['noOfTargetHashtags']:]
        return hashtags_for_lattice
    @staticmethod
    def sharing_probability_with_coverage_distance(propagation_for_prediction, *args, **conf): 
        loadSharingProbabilities()
        coverage_distances_for_hashtags = propagation_for_prediction.getCoverageDistances()
        return PredictionModels._hashtags_by_location_and_coverage_probabilities(propagation_for_prediction, SHARING_PROBABILITIES, coverage_distances_for_hashtags, *args, **conf)
    @staticmethod
    def transmitting_probability_with_coverage_distance(propagation_for_prediction, *args, **conf): 
        loadTransmittingProbabilities()
        coverage_distances_for_hashtags = propagation_for_prediction.getCoverageDistances()
        return PredictionModels._hashtags_by_location_and_coverage_probabilities(propagation_for_prediction, TRANSMITTING_PROBABILITIES, coverage_distances_for_hashtags, *args, **conf)
PREDICTION_MODEL_METHODS = dict([
                                (PredictionModels.RANDOM, PredictionModels.random),
                                (PredictionModels.GREEDY, PredictionModels.greedy),
                                (PredictionModels.SHARING_PROBABILITY, PredictionModels.sharing_probability),
                                (PredictionModels.TRANSMITTING_PROBABILITY, PredictionModels.transmitting_probability),
                                (PredictionModels.COVERAGE_PROBABILITY, PredictionModels.coverage_probability),
                                (PredictionModels.SHARING_PROBABILITY_WITH_COVERAGE, PredictionModels.sharing_probability_with_coverage),
                                (PredictionModels.TRANSMITTING_PROBABILITY_WITH_COVERAGE, PredictionModels.transmitting_probability_with_coverage),
                                (PredictionModels.COVERAGE_DISTANCE, PredictionModels.coverage_distance),
                                (PredictionModels.SHARING_PROBABILITY_WITH_COVERAGE_DISTANCE, PredictionModels.sharing_probability_with_coverage_distance),
                                (PredictionModels.TRANSMITTING_PROBABILITY_WITH_COVERAGE_DISTANCE, PredictionModels.transmitting_probability_with_coverage_distance),
                            ]) 

class Experiments(object):
    def __init__(self, startTime, endTime, outputFolder, predictionModels, evaluationMetrics, *args, **conf):
        self.startTime, self.endTime, self.outputFolder = startTime, endTime, outputFolder
        self.predictionModels, self.evaluationMetrics = predictionModels, evaluationMetrics
        self.historyTimeInterval, self.predictionTimeInterval = conf['historyTimeInterval'], conf['predictionTimeInterval']
        self.conf = conf
#        self.noOfHashtagsList = noOfHashtagsList
        self.noOfHashtagsList = conf.get('noOfHashtagsList', [])
    def _getSerializableConf(self):
        conf_to_return = {}
        for k, v in self.conf.iteritems(): conf_to_return[k]=v
        conf_to_return['historyTimeInterval'] = self.historyTimeInterval.seconds
        conf_to_return['predictionTimeInterval'] = self.predictionTimeInterval.seconds
        return conf_to_return
    def getModelFile(self, modelId): return modelsFolder%self.outputFolder+'%s_%s/%s_%s/%s/%s'%(self.startTime.strftime('%Y-%m-%d'), self.endTime.strftime('%Y-%m-%d'), self.conf['historyTimeInterval'].seconds/60, self.conf['predictionTimeInterval'].seconds/60, self.conf['noOfTargetHashtags'], modelId)
    def run(self):
        currentTime = self.startTime
        timeUnitDelta = timedelta(seconds=TIME_UNIT_IN_SECONDS)
        historicalTimeUnitsMap, predictionTimeUnitsMap = {}, {}
        loadLocationsList()
        print 'Using file: ', timeUnitWithOccurrencesFile%(self.outputFolder, self.startTime.strftime('%Y-%m-%d'), self.endTime.strftime('%Y-%m-%d'))
        timeUnitsToDataMap = dict([(d['tu'], d) for d in iterateJsonFromFile(timeUnitWithOccurrencesFile%(self.outputFolder, self.startTime.strftime('%Y-%m-%d'), self.endTime.strftime('%Y-%m-%d')))])
        for no_of_hashtags in self.noOfHashtagsList:
            for model_id in self.predictionModels:
                self.conf['noOfTargetHashtags'] = no_of_hashtags
                GeneralMethods.runCommand('rm -rf %s'%self.getModelFile(model_id))
#        map(lambda modelId: GeneralMethods.runCommand('rm -rf %s'%self.getModelFile(modelId)), self.predictionModels)
        while currentTime<self.endTime:
            def entry_method():
                print currentTime, self.historyTimeInterval.seconds/60, self.predictionTimeInterval.seconds/60
                currentOccurrences = []
                currentTimeObject = timeUnitsToDataMap.get(time.mktime(currentTime.timetuple()), {})
                if currentTimeObject: currentOccurrences=currentTimeObject['oc']
                for i in range(self.historyTimeInterval.seconds/TIME_UNIT_IN_SECONDS):
                    historicalTimeUnit = currentTime-i*timeUnitDelta
                    if historicalTimeUnit not in historicalTimeUnitsMap: historicalTimeUnitsMap[historicalTimeUnit]=Propagations(historicalTimeUnit, self.historyTimeInterval)
                    historicalTimeUnitsMap[historicalTimeUnit].update(currentOccurrences)
                for i in range(self.predictionTimeInterval.seconds/TIME_UNIT_IN_SECONDS):
                    predictionTimeUnit = currentTime-i*timeUnitDelta
                    if predictionTimeUnit not in predictionTimeUnitsMap: predictionTimeUnitsMap[predictionTimeUnit]=Propagations(predictionTimeUnit, self.predictionTimeInterval)
                    predictionTimeUnitsMap[predictionTimeUnit].update(currentOccurrences)
            entry_method()
            timeUnitForActualPropagation = currentTime-self.predictionTimeInterval
            timeUnitForPropagationForPrediction = timeUnitForActualPropagation-self.historyTimeInterval
            if timeUnitForPropagationForPrediction in historicalTimeUnitsMap and timeUnitForActualPropagation in predictionTimeUnitsMap:
                for noOfTargetHashtags in self.noOfHashtagsList:
                    self.conf['noOfTargetHashtags'] = noOfTargetHashtags
                    for modelId in self.predictionModels:
                        hashtagsForLattice = PREDICTION_MODEL_METHODS[modelId](historicalTimeUnitsMap[timeUnitForPropagationForPrediction], **self.conf)
                        for metric_id in self.evaluationMetrics:
                            scoresPerLattice = EVALUATION_METRIC_METHODS[metric_id](hashtagsForLattice, predictionTimeUnitsMap[timeUnitForActualPropagation], **self.conf)
                            iterationData = {'conf': self._getSerializableConf(), 'tu': GeneralMethods.getEpochFromDateTimeObject(timeUnitForActualPropagation), 'modelId': modelId, 'metricId': metric_id, 'scoresPerLattice': scoresPerLattice}
                            FileIO.writeToFileAsJson(iterationData, self.getModelFile(modelId))
                del historicalTimeUnitsMap[timeUnitForPropagationForPrediction]; del predictionTimeUnitsMap[timeUnitForActualPropagation]
            currentTime+=timeUnitDelta
    def loadExperimentsData(self):
        iteration_results = {}
        model_ids = set(self.predictionModels)
        model_ids.add(PredictionModels.COVERAGE_DISTANCE)
        for model_id in model_ids:
            print 'Loading data for: ', self.getModelFile(model_id)
            for data in FileIO.iterateJsonFromFile(self.getModelFile(model_id)):
                if data['tu'] not in iteration_results: iteration_results[data['tu']] = defaultdict(dict)
                if data['metricId'] in self.evaluationMetrics: iteration_results[data['tu']][model_id][data['metricId']] = data['scoresPerLattice']
        for time_unit in iteration_results:
            for model_id in [PredictionModels.GREEDY, PredictionModels.RANDOM]:
                for metric_id in self.evaluationMetrics:
                    for location in iteration_results[time_unit][PredictionModels.COVERAGE_DISTANCE][EvaluationMetrics.IMPACT]:
                        if metric_id in iteration_results[time_unit][model_id] and location not in iteration_results[time_unit][model_id][metric_id]: 
                            if metric_id==EvaluationMetrics.IMPACT_DIFFERENCE: iteration_results[time_unit][model_id][metric_id][location] = 1.0
                            else: iteration_results[time_unit][model_id][metric_id][location] = 0.0
            
#        for time_unit, results_for_time_unit in iteration_results.iteritems(): map_from_time_unit_to_max_no_of_locations_predictable[time_unit] = len(results_for_time_unit[PredictionModels.COVERAGE_DISTANCE][EvaluationMetrics.IMPACT])
        return iteration_results
    @staticmethod
    def generateDataForVaryingNumberOfHastags(predictionModels, evaluationMetrics, startTime, endTime, outputFolder):
#        noOfHashtagsList=map(lambda i: i*5, range(1,21))
        noOfHashtagsList = [1]+filter(lambda i: i%2==0, range(2,21))
        for i in range(2,7):
#        for i in [2]:
            conf = dict(historyTimeInterval = timedelta(seconds=3*TIME_UNIT_IN_SECONDS), predictionTimeInterval = timedelta(seconds=i*TIME_UNIT_IN_SECONDS), noOfHashtagsList=noOfHashtagsList)
            Experiments(startTime, endTime, outputFolder, predictionModels, evaluationMetrics, **conf).run()
    @staticmethod
    def getImageFileName(metric): return 'images/%s_%s.png'%(inspect.stack()[1][3], metric)
    @staticmethod
    def plotPerformanceForVaryingPredictionTimeIntervals(predictionModels, evaluationMetrics, startTime, endTime, outputFolder):
        predictionTimeIntervals = map(lambda i: i*TIME_UNIT_IN_SECONDS, [2,3,4,5,6])
        for metric in evaluationMetrics:
            evaluationMetrics = [metric]
            data_to_plot_by_model_id = defaultdict(dict)
            for predictionTimeInterval in predictionTimeIntervals:
                conf = dict(historyTimeInterval = timedelta(seconds=1*TIME_UNIT_IN_SECONDS), predictionTimeInterval = timedelta(seconds=predictionTimeInterval), noOfTargetHashtags=5)
                experiments = Experiments(startTime, endTime, outputFolder, predictionModels, evaluationMetrics, **conf)
                for model_id in experiments.predictionModels:
                    iteration_results = experiments.loadIterationData(model_id)
                    metric_values_for_model = defaultdict(list)
                    for _, data_for_model in iteration_results.iteritems():
                        for metric_id, data_for_metric in data_for_model.iteritems():
                            metric_values_for_model[metric_id]+=filter(lambda l: l!=NAN_VALUE, data_for_metric.values())
                    for metric_id in metric_values_for_model: data_to_plot_by_model_id[model_id][predictionTimeInterval] = np.mean(metric_values_for_model[metric_id])
            for model_id, data_to_plot in data_to_plot_by_model_id.iteritems():
                dataX, dataY = zip(*sorted(data_to_plot.iteritems(), key=itemgetter(0)))
                plt.plot(dataX, dataY, label=model_id, lw=2)
            plt.legend()
            plt.savefig(Experiments.getImageFileName(metric))
            plt.clf()
    @staticmethod
    def plotPerformanceForVaryingHistoricalTimeIntervals(predictionModels, evaluationMetrics, startTime, endTime, outputFolder):
        historicalTimeIntervals = map(lambda i: i*TIME_UNIT_IN_SECONDS, [1,2,3,4,5,6])
        for metric in evaluationMetrics:
            evaluationMetrics = [metric]
            data_to_plot_by_model_id = defaultdict(dict)
            for historicalTimeInterval in historicalTimeIntervals:
                conf = dict(historyTimeInterval = timedelta(seconds=historicalTimeInterval), predictionTimeInterval = timedelta(seconds=4*TIME_UNIT_IN_SECONDS), noOfTargetHashtags=5)
                experiments = Experiments(startTime, endTime, outputFolder, predictionModels, evaluationMetrics, **conf)
                for model_id in experiments.predictionModels:
                    iteration_results = experiments.loadIterationData(model_id)
                    metric_values_for_model = defaultdict(list)
                    for _, data_for_model in iteration_results.iteritems():
                        for metric_id, data_for_metric in data_for_model.iteritems():
                            metric_values_for_model[metric_id]+=filter(lambda l: l!=NAN_VALUE, data_for_metric.values())
                    for metric_id in metric_values_for_model: data_to_plot_by_model_id[model_id][historicalTimeInterval] = np.mean(metric_values_for_model[metric_id])
            for model_id, data_to_plot in data_to_plot_by_model_id.iteritems():
                dataX, dataY = zip(*sorted(data_to_plot.iteritems(), key=itemgetter(0)))
                plt.plot(dataX, dataY, label=model_id, lw=2)
            plt.legend()
            plt.savefig(Experiments.getImageFileName(metric))
            plt.clf()
    @staticmethod
    def plotPerformanceForVaryingNoOfHashtags(predictionModels, evaluationMetrics, startTime, endTime, outputFolder):
#        noOfHashtagsList=map(lambda i: i*5, range(1,21))
        noOfHashtagsList=filter(lambda i: i%2==0, range(1,17))
        conf = dict(historyTimeInterval = timedelta(seconds=1*TIME_UNIT_IN_SECONDS), predictionTimeInterval = timedelta(seconds=2*TIME_UNIT_IN_SECONDS), noOfHashtagsList=noOfHashtagsList)
#        for metric in evaluationMetrics:
        experiments = Experiments(startTime, endTime, outputFolder, predictionModels, evaluationMetrics, **conf)
        data_to_plot_by_model_id = defaultdict(dict)
        for noOfTargetHashtags in experiments.noOfHashtagsList:
            experiments.conf['noOfTargetHashtags'] = noOfTargetHashtags
#            for model_id in experiments.predictionModels:
            iteration_results = experiments.loadExperimentsData()
            metric_values_for_model = defaultdict(dict)
#            for model_id in experiments.predictionModels:
            for _, data_for_models in iteration_results.iteritems():
                for model_id in experiments.predictionModels:
                    for metric_id, data_for_metric in data_for_models[model_id].iteritems():
                        if metric_id not in metric_values_for_model[model_id]: metric_values_for_model[model_id][metric_id] = []
                        metric_values_for_model[model_id][metric_id]+=filter(lambda l: l!=NAN_VALUE, data_for_metric.values())
            for model_id in metric_values_for_model: 
                for metric_id in metric_values_for_model[model_id]:
                    if model_id not in data_to_plot_by_model_id[metric_id]: data_to_plot_by_model_id[metric_id][model_id] = {}
                    data_to_plot_by_model_id[metric_id][model_id][noOfTargetHashtags] = np.mean(metric_values_for_model[model_id][metric_id])
#                    for metric_id in metric_values_for_model: data_to_plot_by_model_id[model_id][noOfTargetHashtags] = metric_values_for_model[metric_id]
#        print 'x'
        for metric_id in experiments.evaluationMetrics:
            for model_id, data_to_plot in data_to_plot_by_model_id[metric_id].iteritems():
                dataX, dataY = zip(*sorted(data_to_plot.iteritems(), key=itemgetter(0)))
                plt.plot(dataX, dataY, label=model_id, lw=2)
            plt.legend()
            plt.ylim(ymin=0.0, ymax=1.0)
            plt.savefig(Experiments.getImageFileName(metric_id))
            plt.clf()

#def generateDataForVaryingNoOfHashtagsAtVaryingPredictionTimeInterval(historyTimeInterval, predictionTimeInterval):
#    noOfHashtagsList=map(lambda i: i*5, range(1,21))
#    startTime, endTime, outputFolder = datetime(2011, 11, 1), datetime(2011, 11, 3), 'testing'
#    conf = dict(historyTimeInterval = timedelta(seconds=historyTimeInterval), predictionTimeInterval = timedelta(seconds=predictionTimeInterval), noOfHashtagsList=noOfHashtagsList)
#    predictionModels = [PredictionModels.RANDOM , PredictionModels.GREEDY, PredictionModels.SHARING_PROBABILITY, PredictionModels.TRANSMITTING_PROBABILITY]
#    evaluationMetrics = [EvaluationMetrics.ACCURACY, EvaluationMetrics.IMPACT, EvaluationMetrics.IMPACT_DIFFERENCE]
#    Experiments(startTime, endTime, outputFolder, predictionModels, evaluationMetrics, **conf).run()
        
def temp():
#    d = {}
#    d = [(datetime.fromtimestamp(data['tu']), data['oc']) for e, data in enumerate(iterateJsonFromFile('/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/testing/timeUnitWithOccurrences'))]
#    d = sorted(d, key=itemgetter(0))
#    for t in d: print t[0], len(t[1]), len(set(zip(*t[1])[1]))
#        print e, data.keys()
#        d[data['tu']] = filter(lambda l: l[1] in LOCATIONS_LIST,data['oc'])
#        print datetime.fromtimestamp(data['tu']), len(data['oc'])
    startTime, endTime, outputFolder = datetime(2011, 9, 1), datetime(2012, 12, 31), 'testing'
    for i, data in enumerate(iterateJsonFromFile('/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/testing/2011-09-01_2011-09-16/hashtagsWithEndingWindow')):
        print unicode(data['h']).encode('utf-8'), data['t']
if __name__ == '__main__':
#    loadLocationsList()
#    temp()
#    exit()

#    startTime, endTime, outputFolder = datetime(2011, 9, 1), datetime(2011, 12, 31), 'testing'
    startTime, endTime, outputFolder = datetime(2011, 9, 1), datetime(2011, 11, 1), 'testing'
    predictionModels = [
                        PredictionModels.RANDOM , PredictionModels.GREEDY, 
                        PredictionModels.SHARING_PROBABILITY, PredictionModels.TRANSMITTING_PROBABILITY,
#                        PredictionModels.COVERAGE_PROBABILITY, PredictionModels.SHARING_PROBABILITY_WITH_COVERAGE, PredictionModels.TRANSMITTING_PROBABILITY_WITH_COVERAGE,
                        PredictionModels.COVERAGE_DISTANCE, 
#                        PredictionModels.SHARING_PROBABILITY_WITH_COVERAGE_DISTANCE, PredictionModels.TRANSMITTING_PROBABILITY_WITH_COVERAGE_DISTANCE
                        ]
#    predictionModels = [PredictionModels.RANDOM , PredictionModels.GREEDY]
    evaluationMetrics = [EvaluationMetrics.ACCURACY, EvaluationMetrics.IMPACT, EvaluationMetrics.IMPACT_DIFFERENCE]
    
    Experiments.generateDataForVaryingNumberOfHastags(predictionModels, evaluationMetrics, startTime, endTime, outputFolder)
#    Experiments.plotPerformanceForVaryingNoOfHashtags(predictionModels, evaluationMetrics, startTime, endTime, outputFolder)
#    Experiments.plotPerformanceForVaryingPredictionTimeIntervals(predictionModels, evaluationMetrics, startTime, endTime, outputFolder)
#    Experiments.plotPerformanceForVaryingHistoricalTimeIntervals(predictionModels, evaluationMetrics, startTime, endTime, outputFolder)
    
#    startTime, endTime, outputFolder = datetime(2011, 11, 1), datetime(2011, 12, 1), 'testing'
#    conf = dict(historyTimeInterval = timedelta(seconds=6*TIME_UNIT_IN_SECONDS), 
#                predictionTimeInterval = timedelta(seconds=24*TIME_UNIT_IN_SECONDS),
#                noOfTargetHashtags = 25)
##    
#    predictionModels = [PredictionModels.RANDOM , PredictionModels.GREEDY, PredictionModels.SHARING_PROBABILITY, PredictionModels.TRANSMITTING_PROBABILITY]
##    
#    evaluationMetrics = [EvaluationMetrics.IMPACT_DIFFERENCE]
#    Experiments(startTime, endTime, outputFolder, predictionModels, evaluationMetrics, **conf).plotPerformanceForVaryingNoOfHashtags()
