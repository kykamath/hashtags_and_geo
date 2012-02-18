'''
Created on Feb 14, 2012

@author: kykamath
'''
from analysis import iterateJsonFromFile
from settings import timeUnitWithOccurrencesFile, modelsFile
from datetime import datetime, timedelta
from mr_analysis import TIME_UNIT_IN_SECONDS
import time, random
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from library.classes import GeneralMethods
from library.file_io import FileIO
import numpy as np

NAN_VALUE = -1.0

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
        for loc, hashtags in hashtagsForLocation.iteritems(): metricScorePerLocation[loc] = len(set(hashtags).intersection(set(bestHashtagsForLocation.get(loc, []))))/float(conf['noOfTargetHashtags'])
        return (EvaluationMetrics.ACCURACY, metricScorePerLocation)
    @staticmethod
    def impact(hashtagsForLattice, actualPropagation, *args, **kwargs):
        metricScorePerLocation = {}
        for loc, hashtags in hashtagsForLattice.iteritems(): metricScorePerLocation[loc] = EvaluationMetrics._impact(loc, hashtags, actualPropagation)
        return (EvaluationMetrics.IMPACT, metricScorePerLocation)
    @staticmethod
    def impactDifference(hashtagsForLattice, actualPropagation, *args, **kwargs):
        bestHashtagsForLocation, metricScorePerLocation = EvaluationMetrics._bestHashtagsForLocation(actualPropagation), {}
        for loc, hashtags in hashtagsForLattice.iteritems(): metricScorePerLocation[loc] = EvaluationMetrics._impact(loc, bestHashtagsForLocation.get(loc, []), actualPropagation) - EvaluationMetrics._impact(loc, hashtags, actualPropagation)
        return (EvaluationMetrics.IMPACT_DIFFERENCE, metricScorePerLocation)

class PredictionModels:
    RANDOM = 'random'
    GREEDY = 'greedy'
    @staticmethod
    def random(propagationForPrediction, *args, **conf):
        hashtagsForLattice = defaultdict(list)
        if propagationForPrediction.occurrences:
            for loc, occs in propagationForPrediction.occurrences.iteritems():
                uniqueHashtags = set(zip(*occs)[0])
                hashtagsForLattice[loc] = random.sample(uniqueHashtags, min(len(uniqueHashtags), conf['noOfTargetHashtags']))
        return (PredictionModels.RANDOM, hashtagsForLattice)
    @staticmethod
    def greedy(propagationForPrediction, *args, **conf):
        hashtagsForLattice = defaultdict(list)
        if propagationForPrediction.occurrences:
            for loc, occs in propagationForPrediction.occurrences.iteritems():
                hashtagsForLattice[loc] = zip(*sorted([(h, len(list(hOccs)))for h, hOccs in groupby(sorted(occs, key=itemgetter(0)), key=itemgetter(0))], key=itemgetter(1)))[0][-conf['noOfTargetHashtags']:]
        return (PredictionModels.GREEDY, hashtagsForLattice)
    
class ModelSimulator(object):
    def __init__(self, startTime, endTime, outputFolder, predictionModels, evaluationMetrics, *args, **conf):
        self.startTime, self.endTime, self.outputFolder = startTime, endTime, outputFolder
        self.predictionModels, self.evaluationMetrics = predictionModels, evaluationMetrics
        self.historyTimeInterval, self.predictionTimeInterval = conf['historyTimeInterval'], conf['predictionTimeInterval']
        self.conf = conf
    def run(self):
        currentTime = self.startTime
        timeUnitDelta = timedelta(seconds=TIME_UNIT_IN_SECONDS)
        historicalTimeUnitsMap, predictionTimeUnitsMap = {}, {}
        timeUnitsToDataMap = dict([(d['tu'], d) for d in iterateJsonFromFile(timeUnitWithOccurrencesFile%self.outputFolder)])
        GeneralMethods.runCommand('rm -rf %s'%modelsFile)
        while currentTime<self.endTime:
            print currentTime
            currentOccurrences = {}
            currentTimeObject = timeUnitsToDataMap.get(time.mktime(currentTime.timetuple()), {})
            if currentTimeObject: 
                currentOccurrences=currentTimeObject['oc']
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
                for model in self.predictionModels:
                    modelId, hashtagsForLattice = model(historicalTimeUnitsMap[timeUnitForPropagationForPrediction], **self.conf)
                    for metric in self.evaluationMetrics:
                        metricId, scoresPerLattice = metric(hashtagsForLattice, predictionTimeUnitsMap[timeUnitForActualPropagation], **self.conf)
                        iterationData = {'tu': GeneralMethods.getEpochFromDateTimeObject(timeUnitForActualPropagation), 'modelId': modelId, 'metricId': metricId, 'scoresPerLattice': scoresPerLattice}
                        FileIO.writeToFileAsJson(iterationData, modelsFile)
                del historicalTimeUnitsMap[timeUnitForPropagationForPrediction]; del predictionTimeUnitsMap[timeUnitForActualPropagation]
            currentTime+=timeUnitDelta
    @staticmethod
    def loadIterationData():
        iteration_results = {}
        for data in FileIO.iterateJsonFromFile(modelsFile):
            if data['tu'] not in iteration_results: iteration_results[data['tu']] = {}
            if data['modelId'] not in iteration_results[data['tu']]: iteration_results[data['tu']][data['modelId']] = {}
            iteration_results[data['tu']][data['modelId']][data['metricId']] = data['scoresPerLattice']
        return iteration_results
    @staticmethod
    def plotRunningTimes():
        iteration_results = ModelSimulator.loadIterationData()
        metric_values_for_model = defaultdict(dict)
        for _, data_for_time_unit in iteration_results.iteritems():
            for model_id, data_for_model in data_for_time_unit.iteritems():
                for metric_id, data_for_metric in data_for_model.iteritems():
                    if metric_id not in metric_values_for_model[model_id]: metric_values_for_model[model_id][metric_id] = []
                    metric_values_for_model[model_id][metric_id]+=filter(lambda l: l!=NAN_VALUE, data_for_metric.values())
        for model_id in metric_values_for_model:
            for metric_id in metric_values_for_model[model_id]:
                print model_id, metric_id, np.mean(metric_values_for_model[model_id][metric_id])
#                    print 'x'
#        for data in FileIO.iterateJsonFromFile(modelsFile):
#            print data

if __name__ == '__main__':
    startTime, endTime, outputFolder = datetime(2011, 11, 1), datetime(2011, 12, 31), 'testing'
    conf = dict(historyTimeInterval = timedelta(seconds=30*60), 
                predictionTimeInterval = timedelta(seconds=120*60),
                noOfTargetHashtags = 3)
    
    predictionModels = [PredictionModels.random, PredictionModels.greedy]
    evaluationMetrics = [EvaluationMetrics.accuracy, EvaluationMetrics.impact, EvaluationMetrics.impactDifference]
    
#    ModelSimulator(startTime, endTime, outputFolder, predictionModels, evaluationMetrics, **conf).run()
    ModelSimulator(startTime, endTime, outputFolder, predictionModels, evaluationMetrics, **conf).plotRunningTimes()
