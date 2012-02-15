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
    def update(self, occurrences):
        for h, loc, t in occurrences: self.occurrences[loc].append([h, t])
#        if not testEmptyPropagation(self):
#            print 'there', self.occurrences
#            exit()
            
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
    def accuracy(hashtagsForLattice, actualPropagation, *args, **kwargs):
        bestHashtagsForLattice, metricScorePerLocation = defaultdict(list), {}
        if actualPropagation.occurrences:
            for loc, occs in actualPropagation.occurrences.iteritems():
#                print loc, occs
#                print testEmptyPropagation(actualPropagation)
#                try:
#                if not testEmptyPropagation(actualPropagation):
#                    print actualPropagation.occurrences
#                    exit()
                bestHashtagsForLattice[loc] = zip(*sorted([(h, len(list(hOccs)))for h, hOccs in groupby(sorted(occs, key=itemgetter(0)), key=itemgetter(0))], key=itemgetter(1)))[0][-conf['noOfTargetHashtags']:]
#                except:
#                    pass
            for loc, hashtags in hashtagsForLattice.iteritems(): metricScorePerLocation[loc] = len(set(hashtags).intersection(set(bestHashtagsForLattice[loc])))/float(conf['noOfTargetHashtags'])
        return (EvaluationMetrics.ACCURACY, metricScorePerLocation)
    @staticmethod
    def impact(hashtagsForLattice, actualPropagation, *args, **kwargs):
        metricScorePerLocation = {}
        for loc, hashtags in hashtagsForLattice.iteritems():
            if loc in actualPropagation.occurrences: 
                totalOccs = len(actualPropagation.occurrences[loc])
                occsOfTargetHashtags = len([h for h, t in actualPropagation.occurrences[loc] if h in hashtags])
                metricScorePerLocation[loc] = float(occsOfTargetHashtags)/totalOccs
            else: metricScorePerLocation[loc] = float('nan')
        return (EvaluationMetrics.IMPACT, metricScorePerLocation)
    @staticmethod
    def impactDifference(hashtagsForLattice, actualPropagation, *args, **kwargs):
        bestHashtagsForLattice, metricScorePerLocation = defaultdict(list), {}
            
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
#                if not testEmptyPropagation(predictionTimeUnitsMap[timeUnitForActualPropagation]):
#                            print '1. here', predictionTimeUnitsMap[timeUnitForActualPropagation].occurrences
#                else: print 'clrar'
                for model in self.predictionModels:
#                    if not testEmptyPropagation(predictionTimeUnitsMap[timeUnitForActualPropagation]):
#                            print 'here', predictionTimeUnitsMap[timeUnitForActualPropagation].occurrences
#                            exit() 
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
    
    predictionModels = [PredictionModels.random, PredictionModels.greedy]
    evaluationMetrics = [EvaluationMetrics.accuracy, EvaluationMetrics.impact]
    
    ModelSimulator(startTime, endTime, outputFolder, predictionModels, evaluationMetrics, **conf).run()
