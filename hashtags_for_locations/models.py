'''
Created on Feb 14, 2012

@author: kykamath
'''
from analysis import iterateJsonFromFile
from settings import timeUnitWithOccurrencesFile, locationsGraphFile,\
                     modelsFolder
from datetime import datetime, timedelta
from mr_analysis import TIME_UNIT_IN_SECONDS
import time, random, math
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from library.classes import GeneralMethods, timeit
from library.file_io import FileIO
import numpy as np
from library.stats import getOutliersRangeUsingIRQ

NAN_VALUE = -1.0

LOCATIONS_LIST, SHARING_PROBABILITIES = None, None

def filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeHashtags, neighborHashtags, findLag=True):
    if findLag: 
        dataToReturn = [(hashtag, np.abs(latticeHashtags[hashtag][0]-timeTuple[0])/TIME_UNIT_IN_SECONDS) for hashtag, timeTuple in neighborHashtags.iteritems() if hashtag in latticeHashtags]
        _, upperRangeForTemporalDistance = getOutliersRangeUsingIRQ(zip(*(dataToReturn))[1])
        return dict(filter(lambda t: t[1]<=upperRangeForTemporalDistance, dataToReturn))
    else: 
        dataToReturn = [(hashtag, timeTuple, np.abs(latticeHashtags[hashtag][0]-timeTuple[0])/TIME_UNIT_IN_SECONDS) for hashtag, timeTuple in neighborHashtags.iteritems() if hashtag in latticeHashtags]
        _, upperRangeForTemporalDistance = getOutliersRangeUsingIRQ(zip(*(dataToReturn))[2])
        return dict([(t[0], t[1]) for t in dataToReturn if t[2]<=upperRangeForTemporalDistance])
@timeit
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

def getTransmittingProbabilities():
    model = {'neighborProbability': defaultdict(dict), 'hashtagObservingProbability': {}}
    hashtagsObserved = []
    for latticeObject in FileIO.iterateJsonFromFile(locationsGraphFile):
        latticeHashtagsSet = set(latticeObject['hashtags'])
        hashtagsObserved+=latticeObject['hashtags']
        model['hashtagObservingProbability'][latticeObject['id']] = latticeHashtagsSet
        for neighborLattice, neighborHashtags in latticeObject['links'].iteritems():
            neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags, findLag=False)
#                neighborHashtagsSet = set(neighborHashtags)
            transmittedHashtags = [k for k in neighborHashtags if k in latticeObject['hashtags'] and latticeObject['hashtags'][k][0]<neighborHashtags[k][0]]
            model['neighborProbability'][latticeObject['id']][neighborLattice]=len(transmittedHashtags)/float(len(latticeHashtagsSet))
        model['neighborProbability'][latticeObject['id']][latticeObject['id']]=1.0
    totalNumberOfHashtagsObserved=float(len(set(hashtagsObserved)))
    for lattice in model['hashtagObservingProbability'].keys()[:]: model['hashtagObservingProbability'][lattice] = len(model['hashtagObservingProbability'][lattice])/totalNumberOfHashtagsObserved
        

class Propagations:
    def __init__(self, startTime, interval):
        self.startTime, self.interval = startTime, interval
        self.occurrences = defaultdict(list)
    def update(self, occurrences):
        for h, loc, t in occurrences: self.occurrences[loc].append([h, t])
            
#def testEmptyPropagation(propagation):
#    for loc, occs in propagation.occurrences.iteritems():
#        if not occs: 
##            print propagation.occurrences
##            exit()
#            return False
#    return True
            
class EvaluationMetrics:
    ACCURACY = 'accuracy'
    IMPACT = 'impact'
    IMPACT_DIFFERENCE = 'impact_difference'
    @staticmethod
    def _bestHashtagsForLocation(actualPropagation):
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
        bestHashtagsForLocation, metricScorePerLocation = EvaluationMetrics._bestHashtagsForLocation(actualPropagation), {}
        for loc, hashtags in hashtagsForLocation.iteritems(): 
            bestSet = set(bestHashtagsForLocation.get(loc, []))
            if bestSet: metricScorePerLocation[loc] = len(set(hashtags).intersection(bestSet))/float(len(bestSet))
            else: metricScorePerLocation[loc] = NAN_VALUE
        return metricScorePerLocation
    @staticmethod
    def impact(hashtagsForLattice, actualPropagation, *args, **kwargs):
        metricScorePerLocation = {}
        for loc, hashtags in hashtagsForLattice.iteritems(): metricScorePerLocation[loc] = EvaluationMetrics._impact(loc, hashtags, actualPropagation)
        return metricScorePerLocation
    @staticmethod
    def impactDifference(hashtagsForLattice, actualPropagation, *args, **kwargs):
        bestHashtagsForLocation, metricScorePerLocation = EvaluationMetrics._bestHashtagsForLocation(actualPropagation), {}
        for loc, hashtags in hashtagsForLattice.iteritems(): metricScorePerLocation[loc] = EvaluationMetrics._impact(loc, bestHashtagsForLocation.get(loc, []), actualPropagation) - EvaluationMetrics._impact(loc, hashtags, actualPropagation)
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
    @staticmethod
    def _hashtag_distribution_in_locations(occurrences):
        hashtag_distribution, hashtag_distribution_in_locations = defaultdict(dict), defaultdict(dict)
        for location, occs in occurrences.iteritems():
            for h, _ in occs: 
                if location not in hashtag_distribution[h]: hashtag_distribution[h][location] = 0
                hashtag_distribution[h][location]+=1
        for h in hashtag_distribution.keys()[:]: 
            total_occurrences = float(sum(hashtag_distribution[h].values()))
            for l, v in hashtag_distribution[h].iteritems(): hashtag_distribution_in_locations[l][h] = v/total_occurrences
#            hashtag_distribution[h] = dict([(l, v/float(sum(hashtag_distribution[h].values()))) for l, v in hashtag_distribution[h].iteritems()])
        return hashtag_distribution_in_locations
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
    def sharing_probability(propagation_for_prediction, *args, **conf):
        hashtags_for_lattice = defaultdict(list)
        loadSharingProbabilities()
        hashtag_distribution_in_locations = PredictionModels._hashtag_distribution_in_locations(propagation_for_prediction.occurrences)
        if propagation_for_prediction.occurrences:
            for loc, occs in propagation_for_prediction.occurrences.iteritems():
                hashtag_scores = defaultdict(float)
                for neighboring_location in SHARING_PROBABILITIES['neighborProbability'][loc]:
                    for h in hashtag_distribution_in_locations[loc]: hashtag_scores[h]+=math.log(hashtag_distribution_in_locations[loc][h]) + math.log(SHARING_PROBABILITIES['neighborProbability'][loc][neighboring_location])
#                hashtags_for_lattice[loc] = []
                hashtags_for_lattice[loc] = zip(*sorted([(h, len(list(hOccs)))for h, hOccs in groupby(sorted(occs, key=itemgetter(0)), key=itemgetter(0))], key=itemgetter(1)))[0][-conf['noOfTargetHashtags']:]
                locations = list(zip(*sorted(hashtag_scores.iteritems(), key=itemgetter(1)))[0][-conf['noOfTargetHashtags']:])
                while len(hashtags_for_lattice[loc])<conf['noOfTargetHashtags'] and locations:
                    l = locations.pop()
                    if l not in hashtags_for_lattice[loc]: hashtags_for_lattice[loc].append(l)
        return hashtags_for_lattice
PREDICTION_MODEL_METHODS = dict([(PredictionModels.RANDOM, PredictionModels.random),
                (PredictionModels.GREEDY, PredictionModels.greedy),
                (PredictionModels.SHARING_PROBABILITY, PredictionModels.sharing_probability),
                ])
    
class Experiments(object):
    def __init__(self, startTime, endTime, outputFolder, predictionModels, evaluationMetrics, *args, **conf):
        self.startTime, self.endTime, self.outputFolder = startTime, endTime, outputFolder
        self.predictionModels, self.evaluationMetrics = predictionModels, evaluationMetrics
        self.historyTimeInterval, self.predictionTimeInterval = conf['historyTimeInterval'], conf['predictionTimeInterval']
        self.conf = conf
    def getModelFile(self, modelId): return modelsFolder%self.outputFolder+'%s_%s/%s/%s'%(self.conf['historyTimeInterval'].seconds/60, self.conf['predictionTimeInterval'].seconds/60, self.conf['noOfTargetHashtags'], modelId)
    def run(self):
        currentTime = self.startTime
        timeUnitDelta = timedelta(seconds=TIME_UNIT_IN_SECONDS)
        historicalTimeUnitsMap, predictionTimeUnitsMap = {}, {}
        loadLocationsList()
        timeUnitsToDataMap = dict([(d['tu'], d) for d in iterateJsonFromFile(timeUnitWithOccurrencesFile%self.outputFolder)])
        map(lambda modelId: GeneralMethods.runCommand('rm -rf %s'%self.getModelFile(modelId)), self.predictionModels)
        while currentTime<self.endTime:
            print currentTime
            currentOccurrences = []
            currentTimeObject = timeUnitsToDataMap.get(time.mktime(currentTime.timetuple()), {})
            if currentTimeObject: currentOccurrences=filter(lambda l: l[1] in LOCATIONS_LIST,currentTimeObject['oc'])
            for i in range(self.historyTimeInterval.seconds/TIME_UNIT_IN_SECONDS):
                historicalTimeUnit = currentTime-i*timeUnitDelta
                if historicalTimeUnit not in historicalTimeUnitsMap: historicalTimeUnitsMap[historicalTimeUnit]=Propagations(historicalTimeUnit, self.historyTimeInterval)
                historicalTimeUnitsMap[historicalTimeUnit].update(currentOccurrences)
            for i in range(self.predictionTimeInterval.seconds/TIME_UNIT_IN_SECONDS):
                predictionTimeUnit = currentTime-i*timeUnitDelta
                if predictionTimeUnit not in predictionTimeUnitsMap: predictionTimeUnitsMap[predictionTimeUnit]=Propagations(predictionTimeUnit, self.predictionTimeInterval)
                predictionTimeUnitsMap[predictionTimeUnit].update(currentOccurrences)
            timeUnitForActualPropagation = currentTime-self.predictionTimeInterval
            timeUnitForPropagationForPrediction = timeUnitForActualPropagation-self.historyTimeInterval
            if timeUnitForPropagationForPrediction in historicalTimeUnitsMap and timeUnitForActualPropagation in predictionTimeUnitsMap:
                print currentTime, len(historicalTimeUnitsMap[timeUnitForPropagationForPrediction].occurrences), len(predictionTimeUnitsMap[timeUnitForActualPropagation].occurrences)
                continue
                for modelId in self.predictionModels:
                    hashtagsForLattice = PREDICTION_MODEL_METHODS[modelId](historicalTimeUnitsMap[timeUnitForPropagationForPrediction], **self.conf)
                    for metric_id in self.evaluationMetrics:
                        scoresPerLattice = EVALUATION_METRIC_METHODS[metric_id](hashtagsForLattice, predictionTimeUnitsMap[timeUnitForActualPropagation], **self.conf)
                        iterationData = {'tu': GeneralMethods.getEpochFromDateTimeObject(timeUnitForActualPropagation), 'modelId': modelId, 'metricId': metric_id, 'scoresPerLattice': scoresPerLattice}
                        FileIO.writeToFileAsJson(iterationData, self.getModelFile(modelId))
                del historicalTimeUnitsMap[timeUnitForPropagationForPrediction]; del predictionTimeUnitsMap[timeUnitForActualPropagation]
            currentTime+=timeUnitDelta
    def loadIterationData(self, modelId):
        iteration_results = {}
        for data in FileIO.iterateJsonFromFile(self.getModelFile(modelId)):
            if data['tu'] not in iteration_results: iteration_results[data['tu']] = {}
#            if data['modelId'] not in iteration_results[data['tu']]: iteration_results[data['tu']][data['modelId']] = {}
            if data['metricId'] in self.evaluationMetrics: iteration_results[data['tu']][data['metricId']] = data['scoresPerLattice']
        return iteration_results
    def plotRunningTimes(self):
        for model_id in self.predictionModels:
            iteration_results = self.loadIterationData(model_id)
            metric_values_for_model = defaultdict(list)
            for _, data_for_model in iteration_results.iteritems():
#                for model_id, data_for_model in data_for_time_unit.iteritems():
                for metric_id, data_for_metric in data_for_model.iteritems():
#                    if metric_id not in metric_values_for_model[model_id]: metric_values_for_model[metric_id] = []
                    metric_values_for_model[metric_id]+=filter(lambda l: l!=NAN_VALUE, data_for_metric.values())
#            for model_id in metric_values_for_model:
            for metric_id in metric_values_for_model:
                print model_id, metric_id, np.mean(metric_values_for_model[metric_id])
@timeit
def temp():
    d = {}
    d = [(datetime.fromtimestamp(data['tu']), data['oc']) for e, data in enumerate(iterateJsonFromFile('/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/testing/timeUnitWithOccurrences'))]
    d = sorted(d, key=itemgetter(0))
    for t in d: print t[0], len(t[1]), len(set(zip(*t[1])[1]))
#        print e, data.keys()
#        d[data['tu']] = filter(lambda l: l[1] in LOCATIONS_LIST,data['oc'])
#        print datetime.fromtimestamp(data['tu']), len(data['oc'])
if __name__ == '__main__':
#    loadLocationsList()
    temp()
    exit()
    
    startTime, endTime, outputFolder = datetime(2011, 9, 1), datetime(2011, 12, 31), 'testing'
    conf = dict(historyTimeInterval = timedelta(seconds=6*TIME_UNIT_IN_SECONDS), 
                predictionTimeInterval = timedelta(seconds=24*TIME_UNIT_IN_SECONDS),
                noOfTargetHashtags = 10)
    
    predictionModels = [PredictionModels.RANDOM , PredictionModels.GREEDY, PredictionModels.SHARING_PROBABILITY]
    
    evaluationMetrics = [EvaluationMetrics.ACCURACY, EvaluationMetrics.IMPACT, EvaluationMetrics.IMPACT_DIFFERENCE]
#    evaluationMetrics = [EvaluationMetrics.IMPACT_DIFFERENCE]
    
    Experiments(startTime, endTime, outputFolder, predictionModels, evaluationMetrics, **conf).run()
#    Experiments(startTime, endTime, outputFolder, predictionModels, evaluationMetrics, **conf).plotRunningTimes()
