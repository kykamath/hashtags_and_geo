'''
Created on Feb 14, 2012

@author: kykamath
'''
from analysis import iterateJsonFromFile
from settings import timeUnitWithOccurrencesFile
from datetime import datetime, timedelta
from mr_analysis import TIME_UNIT_IN_SECONDS
import time
from collections import defaultdict

historyTimeInterval = timedelta(seconds=30*60)
predictionTimeInterval = timedelta(seconds=120*60)

class Propagations:
    def __init__(self, startTime, interval):
        self.startTime, self.interval = startTime, interval
        self.occurrances = defaultdict(list)
    def update(self, occurrances):
        for h, loc, t in occurrances: 
            self.occurrances[h].append([loc, t])
            
class EvaluationMetrics:
    pass
            
class PredictionModels:
    RANDOM = 'random'
    @staticmethod
    def random(propagationForPrediction, actualPropagation):
        print propagationForPrediction.startTime, propagationForPrediction.interval
        print actualPropagation.startTime, actualPropagation.interval
    
class ModelSimulator(object):
    def __init__(self, startTime, endTime, outputFolder, predictionModels, *args, **kwargs):
        self.startTime, self.endTime, self.outputFolder = startTime, endTime, outputFolder
        self.predictionModels = predictionModels
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
            for i in range(historyTimeInterval.seconds/TIME_UNIT_IN_SECONDS):
                historicalTimeUnit = currentTime-i*timeUnitDelta
                if historicalTimeUnit not in historicalTimeUnitsMap: historicalTimeUnitsMap[historicalTimeUnit]=Propagations(historicalTimeUnit, historyTimeInterval)
                historicalTimeUnitsMap[historicalTimeUnit].update(currentOccurrences)
            for i in range(predictionTimeInterval.seconds/TIME_UNIT_IN_SECONDS):
                predictionTimeUnit = currentTime-i*timeUnitDelta
                if predictionTimeUnit not in predictionTimeUnitsMap: predictionTimeUnitsMap[predictionTimeUnit]=Propagations(predictionTimeUnit, predictionTimeInterval)
                predictionTimeUnitsMap[predictionTimeUnit].update(currentOccurrences)
            timeUnitForActualPropagation = currentTime-predictionTimeInterval
            timeUnitForPropagationForPrediction = timeUnitForActualPropagation-historyTimeInterval
            if timeUnitForPropagationForPrediction in historicalTimeUnitsMap and timeUnitForActualPropagation in predictionTimeUnitsMap:
                for model in self.predictionModels: print model(historicalTimeUnitsMap[timeUnitForPropagationForPrediction], predictionTimeUnitsMap[timeUnitForActualPropagation])
                del historicalTimeUnitsMap[timeUnitForPropagationForPrediction]; del predictionTimeUnitsMap[timeUnitForActualPropagation]
            currentTime+=timeUnitDelta
    
#    def process(self, propagationForPrediction, actualPropagation):
#        print propagationForPrediction.startTime, propagationForPrediction.interval
#        print actualPropagation.startTime, actualPropagation.interval

if __name__ == '__main__':
    startTime, endTime, outputFolder = datetime(2011, 11, 1), datetime(2011, 12, 31), 'testing'
    
    predictionModels = [PredictionModels.random]
    
    ModelSimulator(startTime, endTime, outputFolder, predictionModels).run()
#for k, v in predictionTimeUnitsMap.iteritems():
#    print k, v.count
    
#timeUnitsToDataMap = [(d['tu'], d) for d in iterateJsonFromFile(timeUnitWithOccurrencesFile%outputFolder)]
#for i, data in enumerate(sorted(data, key=lambda d: d['tu'])):
#    print i, data['tu'], datetime.fromtimestamp(data['tu'])