'''
Created on Feb 14, 2012

@author: kykamath
'''
from analysis import iterateJsonFromFile
from settings import timeUnitWithOccurrencesFile
from datetime import datetime, timedelta
from mr_analysis import TIME_UNIT_IN_SECONDS
import time, random
from collections import defaultdict
from itertools import groupby
from operator import itemgetter

class Propagations:
    def __init__(self, startTime, interval):
        self.startTime, self.interval = startTime, interval
        self.occurrences = defaultdict(list)
    def update(self, occurrances):
        for h, loc, t in occurrances: 
            self.occurrences[loc].append([h, t])
            
class EvaluationMetrics:
    ACCURACY = 'accuracy'
    @staticmethod
    def accuracy(hashtagsForLattice, actualPropagation, *args, **kwargs):
        return (EvaluationMetrics.ACCURACY, 1.0)
            
class PredictionModels:
    RANDOM = 'random'
    @staticmethod
    def random(propagationForPrediction, *args, **conf):
        hashtagsForLattice = defaultdict(list)
        if propagationForPrediction.occurrences:
            for loc, occs in propagationForPrediction.occurrences.iteritems():
                uniqueHashtags = set(zip(*occs)[0])
                hashtagsForLattice[loc] = random.sample(uniqueHashtags, min(len(uniqueHashtags), conf['noOfTargetHashtags']))
#                hashtagsForLattice[loc] = sorted([(h, len(list(hOccs)))for h, hOccs in groupby(sorted(occs, key=itemgetter(0)), key=itemgetter(0))], key=itemgetter(0))[-conf['noOfTargetHashtags']:]
        return (PredictionModels.RANDOM, hashtagsForLattice)
    
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
        timeUnitsToDataMap = dict([(d['tu'], d) for d in iterateJsonFromFile(timeUnitWithOccurrencesFile%outputFolder)])
        while currentTime<self.endTime:
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
                        print modelId, metricId, scoresPerLattice
                del historicalTimeUnitsMap[timeUnitForPropagationForPrediction]; del predictionTimeUnitsMap[timeUnitForActualPropagation]
            currentTime+=timeUnitDelta

if __name__ == '__main__':
    startTime, endTime, outputFolder = datetime(2011, 11, 1), datetime(2011, 12, 31), 'testing'
    conf = dict(historyTimeInterval = timedelta(seconds=30*60), 
                predictionTimeInterval = timedelta(seconds=120*60),
                noOfTargetHashtags = 3)
    
    predictionModels = [PredictionModels.random]
    evaluationMetrics = [EvaluationMetrics.accuracy]
    
    ModelSimulator(startTime, endTime, outputFolder, predictionModels, evaluationMetrics, **conf).run()
#for k, v in predictionTimeUnitsMap.iteritems():
#    print k, v.count
    
#timeUnitsToDataMap = [(d['tu'], d) for d in iterateJsonFromFile(timeUnitWithOccurrencesFile%outputFolder)]
#for i, data in enumerate(sorted(data, key=lambda d: d['tu'])):
#    print i, data['tu'], datetime.fromtimestamp(data['tu'])