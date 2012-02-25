'''
Created on Feb 14, 2012

@author: kykamath
'''
from analysis import iterateJsonFromFile
from settings import timeUnitWithOccurrencesFile, locationsGraphFile,\
                     modelsFolder, hashtagsAllOccurrencesWithinWindowFile
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
import matplotlib.pyplot as plt
from multiprocessing import Pool

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

class Propagations:
    def __init__(self, startTime, interval):
        self.startTime, self.interval = startTime, interval
        self.occurrences = defaultdict(list)
    def update(self, occurrences):
        for h, loc, t in occurrences: self.occurrences[loc].append([h, t])

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
            else: metricScorePerLocation[loc] = NAN_VALUE
        return metricScorePerLocation
    @staticmethod
    def impact(hashtagsForLattice, actualPropagation, *args, **kwargs):
        metricScorePerLocation = {}
        for loc, hashtags in hashtagsForLattice.iteritems(): metricScorePerLocation[loc] = EvaluationMetrics._impact(loc, hashtags, actualPropagation)
        return metricScorePerLocation
    @staticmethod
    def impactDifference(hashtagsForLattice, actualPropagation, *args, **kwargs):
        bestHashtagsForLocation, metricScorePerLocation = EvaluationMetrics._bestHashtagsForLocation(actualPropagation, **kwargs), {}
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
    TRANSMITTING_PROBABILITY = 'tranmitting_probability'
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
    def _hashtags_by_location_probabilities(propagation_for_prediction, location_probabilities, *args, **conf):
        hashtags_for_lattice = defaultdict(list)
        hashtag_distribution_in_locations = PredictionModels._hashtag_distribution_in_locations(propagation_for_prediction.occurrences)
        if propagation_for_prediction.occurrences:
            for loc, occs in propagation_for_prediction.occurrences.iteritems():
                hashtag_scores, hashtags = defaultdict(float), []
                for neighboring_location in location_probabilities['neighborProbability'][loc]:
                    if location_probabilities['neighborProbability'][loc][neighboring_location]!=0.0:
    #                    for h in hashtag_distribution_in_locations[loc]: hashtag_scores[h]+=math.log(hashtag_distribution_in_locations[loc][h]) + math.log(SHARING_PROBABILITIES['neighborProbability'][loc][neighboring_location])
                        for h in hashtag_distribution_in_locations[neighboring_location]: hashtag_scores[h]+=math.log(hashtag_distribution_in_locations[neighboring_location][h]) + math.log(location_probabilities['neighborProbability'][loc][neighboring_location])
#                hashtags_for_lattice[loc] = []
                hashtags_for_lattice[loc] = list(zip(*sorted([(h, len(list(hOccs)))for h, hOccs in groupby(sorted(occs, key=itemgetter(0)), key=itemgetter(0))], key=itemgetter(1)))[0][-conf['noOfTargetHashtags']:])
                if hashtag_scores: hashtags = list(zip(*sorted(hashtag_scores.iteritems(), key=itemgetter(1)))[0][-conf['noOfTargetHashtags']:])
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
PREDICTION_MODEL_METHODS = dict([(PredictionModels.RANDOM, PredictionModels.random),
                (PredictionModels.GREEDY, PredictionModels.greedy),
                (PredictionModels.SHARING_PROBABILITY, PredictionModels.sharing_probability),
                (PredictionModels.TRANSMITTING_PROBABILITY, PredictionModels.transmitting_probability),
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
    def loadIterationData(self, modelId):
        iteration_results = {}
        for data in FileIO.iterateJsonFromFile(self.getModelFile(modelId)):
            if data['tu'] not in iteration_results: iteration_results[data['tu']] = {}
            if data['metricId'] in self.evaluationMetrics: iteration_results[data['tu']][data['metricId']] = data['scoresPerLattice']
        return iteration_results
    @staticmethod
    def generateDataForVaryingNumberOfHastags():
        noOfHashtagsList=map(lambda i: i*5, range(1,21))
        startTime, endTime, outputFolder = datetime(2011, 9, 1), datetime(2012, 12, 31), 'testing'
        conf = dict(historyTimeInterval = timedelta(seconds=1*TIME_UNIT_IN_SECONDS), predictionTimeInterval = timedelta(seconds=2*TIME_UNIT_IN_SECONDS), noOfHashtagsList=noOfHashtagsList)
        predictionModels = [PredictionModels.RANDOM , PredictionModels.GREEDY, PredictionModels.SHARING_PROBABILITY, PredictionModels.TRANSMITTING_PROBABILITY]
        evaluationMetrics = [EvaluationMetrics.ACCURACY, EvaluationMetrics.IMPACT, EvaluationMetrics.IMPACT_DIFFERENCE]
        Experiments(startTime, endTime, outputFolder, predictionModels, evaluationMetrics, **conf).run()
#        po = Pool()
#        po.map_async(generateDataForVaryingNoOfHashtagsAtVaryingPredictionTimeInterval, ((1*TIME_UNIT_IN_SECONDS, i*2*TIME_UNIT_IN_SECONDS) for i in xrange(1,2)))
#        po.close()
#        po.join()

    @staticmethod
    def plotPerformanceForVaryingPredictionTimeIntervals(metric):
        predictionTimeIntervals = map(lambda i: i*TIME_UNIT_IN_SECONDS, [2,4,6,8])
        evaluationMetrics = [metric]
        data_to_plot_by_model_id = defaultdict(dict)
        for predictionTimeInterval in predictionTimeIntervals:
            startTime, endTime, outputFolder = datetime(2011, 11, 1), datetime(2012, 1, 31), 'testing'
            conf = dict(historyTimeInterval = timedelta(seconds=1*TIME_UNIT_IN_SECONDS), predictionTimeInterval = timedelta(seconds=predictionTimeInterval), noOfTargetHashtags=25)
            predictionModels = [PredictionModels.RANDOM , PredictionModels.GREEDY, PredictionModels.SHARING_PROBABILITY, PredictionModels.TRANSMITTING_PROBABILITY]
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
        plt.savefig('images/plotPerformanceForVaryingPredictionTimeIntervals.png')
    @staticmethod
    def plotPerformanceForVaryingHistoricalTimeIntervals(metric):
        historicalTimeIntervals = map(lambda i: i*TIME_UNIT_IN_SECONDS, [1,2])
        evaluationMetrics = [metric]
        data_to_plot_by_model_id = defaultdict(dict)
        for historicalTimeInterval in historicalTimeIntervals:
            startTime, endTime, outputFolder = datetime(2011, 11, 1), datetime(2011, 11, 3), 'testing'
            conf = dict(historyTimeInterval = timedelta(seconds=historicalTimeInterval), predictionTimeInterval = timedelta(seconds=4*TIME_UNIT_IN_SECONDS), noOfTargetHashtags=25)
            predictionModels = [PredictionModels.RANDOM , PredictionModels.GREEDY, PredictionModels.SHARING_PROBABILITY, PredictionModels.TRANSMITTING_PROBABILITY]
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
        plt.show()
#        plt.savefig('images/plotPerformanceForVaryingPredictionTimeIntervals.png')
    @staticmethod
    def plotPerformanceForVaryingNoOfHashtags(metric):
        noOfHashtagsList=map(lambda i: i*5, range(1,21))
        startTime, endTime, outputFolder = datetime(2011, 11, 1), datetime(2012, 1, 31), 'testing'
        conf = dict(historyTimeInterval = timedelta(seconds=1*TIME_UNIT_IN_SECONDS), predictionTimeInterval = timedelta(seconds=4*TIME_UNIT_IN_SECONDS), noOfHashtagsList=noOfHashtagsList)
        predictionModels = [PredictionModels.RANDOM , PredictionModels.GREEDY, PredictionModels.SHARING_PROBABILITY, PredictionModels.TRANSMITTING_PROBABILITY]
        evaluationMetrics = [metric]
        experiments = Experiments(startTime, endTime, outputFolder, predictionModels, evaluationMetrics, **conf)
        data_to_plot_by_model_id = defaultdict(dict)
        for noOfTargetHashtags in experiments.noOfHashtagsList:
            experiments.conf['noOfTargetHashtags'] = noOfTargetHashtags
            for model_id in experiments.predictionModels:
                iteration_results = experiments.loadIterationData(model_id)
                metric_values_for_model = defaultdict(list)
                for _, data_for_model in iteration_results.iteritems():
                    for metric_id, data_for_metric in data_for_model.iteritems():
                        metric_values_for_model[metric_id]+=filter(lambda l: l!=NAN_VALUE, data_for_metric.values())
                for metric_id in metric_values_for_model: data_to_plot_by_model_id[model_id][noOfTargetHashtags] = np.mean(metric_values_for_model[metric_id])
        for model_id, data_to_plot in data_to_plot_by_model_id.iteritems():
            dataX, dataY = zip(*sorted(data_to_plot.iteritems(), key=itemgetter(0)))
            plt.plot(dataX, dataY, label=model_id, lw=2)
        plt.legend()
        plt.savefig('images/plotPerformanceForVaryingNoOfHashtags.png')

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
    for i, data in enumerate(iterateJsonFromFile('/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/testing/2011-09-01_2012-12-31/thashtagsWithEndingWindow')):
        print unicode(data['h']).encode('utf-8'), data['t']
if __name__ == '__main__':
#    loadLocationsList()
#    temp()
#    exit()

    Experiments.generateDataForVaryingNumberOfHastags()
#    Experiments.plotPerformanceForVaryingPredictionTimeIntervals(EvaluationMetrics.IMPACT_DIFFERENCE)
    
#    startTime, endTime, outputFolder = datetime(2011, 11, 1), datetime(2011, 12, 1), 'testing'
#    conf = dict(historyTimeInterval = timedelta(seconds=6*TIME_UNIT_IN_SECONDS), 
#                predictionTimeInterval = timedelta(seconds=24*TIME_UNIT_IN_SECONDS),
#                noOfTargetHashtags = 25)
##    
#    predictionModels = [PredictionModels.RANDOM , PredictionModels.GREEDY, PredictionModels.SHARING_PROBABILITY, PredictionModels.TRANSMITTING_PROBABILITY]
##    
#    evaluationMetrics = [EvaluationMetrics.IMPACT_DIFFERENCE]
#    Experiments(startTime, endTime, outputFolder, predictionModels, evaluationMetrics, **conf).plotPerformanceForVaryingNoOfHashtags()
