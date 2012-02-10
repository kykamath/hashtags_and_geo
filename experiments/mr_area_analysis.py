'''
Created on Nov 19, 2011

@author: kykamath
'''
from library.twitter import getDateTimeObjectFromTweetTimestamp
from library.mrjobwrapper import ModifiedMRJob
from library.geo import getLatticeLid, getLattice, isWithinBoundingBox,\
    getLocationFromLid, getHaversineDistance, getCenterOfMass
import cjson, time, datetime
from collections import defaultdict
from itertools import groupby
import numpy as np
from library.classes import GeneralMethods
from itertools import combinations
from operator import itemgetter
from library.stats import getOutliersRangeUsingIRQ

# General parameters
LATTICE_ACCURACY = 0.145
TIME_UNIT_IN_SECONDS = 60*60

## Paramters for local run
## Paramters to filter hashtags.
#MIN_HASHTAG_OCCURENCES = 1
#HASHTAG_STARTING_WINDOW = time.mktime(datetime.datetime(2011, 1, 1).timetuple())
#
## Paramters to construct lattice graph.
#MIN_NO_OF_TIME_UNITS_IN_INACTIVE_REGION = 1
#MIN_COMMON_HASHTAG_OCCURENCES_BETWEEN_LATTICE_PAIRS = 3
#MIN_UNIQUE_HASHTAG_OCCURENCES_PER_LATTICE = 1
#MIN_HASHTAG_OCCURENCES_PER_LATTICE = 1

# Paramters to filter hashtags.
MIN_HASHTAG_OCCURENCES = 250
#MIN_HASHTAG_OCCURENCES = 1000
#HASHTAG_STARTING_WINDOW = time.mktime(datetime.datetime(2011, 2, 25).timetuple())
#HASHTAG_ENDING_WINDOW = time.mktime(datetime.datetime(2011, 8, 31).timetuple())
#HASHTAG_STARTING_WINDOW = time.mktime(datetime.datetime(2011, 9, 1).timetuple())
#HASHTAG_ENDING_WINDOW = time.mktime(datetime.datetime(2011, 10, 31).timetuple())
#HASHTAG_STARTING_WINDOW = time.mktime(datetime.datetime(2011, 2, 1).timetuple())
#HASHTAG_ENDING_WINDOW = time.mktime(datetime.datetime(2011, 11, 30).timetuple())
HASHTAG_STARTING_WINDOW = time.mktime(datetime.datetime(2011, 3, 1).timetuple())
HASHTAG_ENDING_WINDOW = time.mktime(datetime.datetime(2011, 10, 31).timetuple())

# Paramters to construct lattice graph.
MIN_NO_OF_TIME_UNITS_IN_INACTIVE_REGION = 12
#MIN_COMMON_HASHTAG_OCCURENCES_BETWEEN_LATTICE_PAIRS = 25
#MIN_UNIQUE_HASHTAG_OCCURENCES_PER_LATTICE = 25 # Min no. of unique hashtags a lattice should have observed. For example: l1 is valid of it produces [h1, h2, h3] >= 3 (min)
#MIN_HASHTAG_OCCURENCES_PER_LATTICE = 10 # Min no. hashtags lattice should have observed. For example: l1 is valid of it produces [h1, h1, h1] >= 3 (min)

MIN_COMMON_HASHTAG_OCCURENCES_BETWEEN_LATTICE_PAIRS = 5
MIN_UNIQUE_HASHTAG_OCCURENCES_PER_LATTICE = 5 # Min no. of unique hashtags a lattice should have observed. For example: l1 is valid of it produces [h1, h2, h3] >= 3 (min)
MIN_HASHTAG_OCCURENCES_PER_LATTICE = 5 # Min no. hashtags lattice should have observed. For example: l1 is valid of it produces [h1, h1, h1] >= 3 (min)

NO_OF_EARLY_LIDS_TO_DETERMINE_SOURCE_LATTICE = 10
MIN_OCCURRENCES_TO_DETERMINE_SOURCE_LATTICE = 5

#MIN_TEMPORAL_CLOSENESS_SCORE_FOR_IN_OUT_LINKS = 0.0
#PERCENTAGE_OF_EARLY_LIDS_TO_DETERMINE_SOURCE_LATTICE = 0.01
#MIN_TEMPORAL_CLOSENESS_SCORE = 0.1
#MIN_OBSERVATIONS_GREATER_THAN_MIN_TEMPORAL_CLOSENESS_SCORE = 3

AREAS = {
#         'ny': dict(boundary=[[40.491, -74.356], [41.181, -72.612]]), # New York
#         'north_cal': dict(boundary=[[37.068328,-122.640381], [37.924701,-122.178955]]), # North Cal
#         'austin': dict(boundary=[[30.097613,-97.971954], [30.486551,-97.535248]]),
#         'dallas': dict(boundary=[[32.735307,-96.862335], [32.886507,-96.723633]]),
#         'us': dict(boundary=[[24.527135,-127.792969], [49.61071,-59.765625]]),
        'world': dict(boundary=[[-90,-180], [90, 180]]),  
         }

BOUNDARIES  = [v['boundary'] for v in AREAS.itervalues()]

def iterateHashtagObjectInstances(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    point = getLattice(l, LATTICE_ACCURACY)
#    if isWithinBoundingBox(point, BOUNDING_BOX):
    for h in data['h']: yield h.lower(), [point, t]

#def latticeIdInValidAreas(latticeId):
#    return isWithinBoundingBox(getLocationFromLid(latticeId.replace('_', ' ')), BOUNDING_BOX)

def latticeIdInValidAreas(latticeId):
    point = getLocationFromLid(latticeId.replace('_', ' '))
    for boundary in BOUNDARIES:
        if isWithinBoundingBox(point, boundary): return True

def getHashtagWithoutEndingWindow(key, values):
    occurences = []
    for instances in values: 
        for oc in instances['oc']: occurences.append(oc)
    if occurences:
        e, l = min(occurences, key=lambda t: t[1]), max(occurences, key=lambda t: t[1])
        numberOfInstances=len(occurences)
        if numberOfInstances>=MIN_HASHTAG_OCCURENCES and \
            e[1]>=HASHTAG_STARTING_WINDOW: return {'h': key, 't': numberOfInstances, 'e':e, 'l':l, 'oc': sorted(occurences, key=lambda t: t[1])}

def getHashtagWithEndingWindow(key, values):
    occurences = []
    for instances in values: 
        for oc in instances['oc']: occurences.append(oc)
    if occurences:
        e, l = min(occurences, key=lambda t: t[1]), max(occurences, key=lambda t: t[1])
        numberOfInstances=len(occurences)
        if numberOfInstances>=MIN_HASHTAG_OCCURENCES and \
            e[1]>=HASHTAG_STARTING_WINDOW and l[1]<=HASHTAG_ENDING_WINDOW: return {'h': key, 't': numberOfInstances, 'e':e, 'l':l, 'oc': sorted(occurences, key=lambda t: t[1])}

#def getOccurranceDistributionInEpochs(occ): return [(k[0], len(list(k[1]))) for k in groupby(sorted([GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) for t in zip(*occ)[1]]))]
def getOccurranceDistributionInEpochs(occ, timeUnit=TIME_UNIT_IN_SECONDS, fillInGaps=False, occurancesCount=True): 
    if occurancesCount: occurranceDistributionInEpochs = filter(lambda t:t[1]>2, [(k[0], len(list(k[1]))) for k in groupby(sorted([GeneralMethods.approximateEpoch(t, timeUnit) for t in zip(*occ)[1]]))])
    else: occurranceDistributionInEpochs = filter(lambda t:len(t[1])>2, [(k[0], [t[1] for t in k[1]]) for k in groupby(sorted([(GeneralMethods.approximateEpoch(t[1], timeUnit), t) for t in occ], key=itemgetter(0)), key=itemgetter(0))])
    if not fillInGaps: return occurranceDistributionInEpochs
    else:
        if occurranceDistributionInEpochs:
            startEpoch, endEpoch = min(occurranceDistributionInEpochs, key=itemgetter(0))[0], max(occurranceDistributionInEpochs, key=itemgetter(0))[0]
#            if not occurancesCount: startEpoch, endEpoch = startEpoch[0], endEpoch[0]
            dataX = range(startEpoch, endEpoch, timeUnit)
            occurranceDistributionInEpochs = dict(occurranceDistributionInEpochs)
            for x in dataX: 
                if x not in occurranceDistributionInEpochs: 
                    if occurancesCount: occurranceDistributionInEpochs[x]=0
                    else: occurranceDistributionInEpochs[x]=[]
            return occurranceDistributionInEpochs
        else: return dict(occurranceDistributionInEpochs)
    
    
def getActiveRegions(timeSeries):
    noOfZerosObserved, activeRegions = 0, []
    currentRegion, occurancesForRegion = None, 0
    for index, l in zip(range(len(timeSeries)),timeSeries):
        if l>0: 
            if noOfZerosObserved>MIN_NO_OF_TIME_UNITS_IN_INACTIVE_REGION or index==0:
                currentRegion = [None, None, None]
                currentRegion[0] = index
                occurancesForRegion = 0
            noOfZerosObserved = 0
            occurancesForRegion+=l
        else: 
            noOfZerosObserved+=1
            if noOfZerosObserved>MIN_NO_OF_TIME_UNITS_IN_INACTIVE_REGION and currentRegion and currentRegion[1]==None:
                currentRegion[1] = index-MIN_NO_OF_TIME_UNITS_IN_INACTIVE_REGION-1
                currentRegion[2] = occurancesForRegion
                activeRegions.append(currentRegion)
    if not activeRegions: activeRegions.append([0, len(timeSeries)-1, sum(timeSeries)])
    else: 
        currentRegion[1], currentRegion[2] = index, occurancesForRegion
        activeRegions.append(currentRegion)
    return activeRegions
def getOccuranesInHighestActiveRegion(hashtagObject, checkIfItFirstActiveRegion=False, timeUnit=TIME_UNIT_IN_SECONDS, maxLengthOfHighestActiveRegion=None):
    occurancesInActiveRegion, timeUnits = [], []
    occurranceDistributionInEpochs = getOccurranceDistributionInEpochs(hashtagObject['oc'], fillInGaps=True)
    if occurranceDistributionInEpochs:
        timeUnits, timeSeries = zip(*sorted(occurranceDistributionInEpochs.iteritems(), key=itemgetter(0)))
        hashtagPropagatingRegion = max(getActiveRegions(timeSeries), key=itemgetter(2))
        if not maxLengthOfHighestActiveRegion: validTimeUnits = [timeUnits[i] for i in range(hashtagPropagatingRegion[0], hashtagPropagatingRegion[1]+1)]
        else: validTimeUnits = [timeUnits[i] for i in range(hashtagPropagatingRegion[0], hashtagPropagatingRegion[1]+1)][:maxLengthOfHighestActiveRegion]
        occurancesInActiveRegion = [(p,t) for p,t in hashtagObject['oc'] if GeneralMethods.approximateEpoch(t, timeUnit) in validTimeUnits]
    if not checkIfItFirstActiveRegion: return occurancesInActiveRegion
    else:
        isFirstActiveRegion=False
        if timeUnits and timeUnits[0]==validTimeUnits[0]: isFirstActiveRegion=True
        return (occurancesInActiveRegion, isFirstActiveRegion)
        
def filterLatticesByMinHashtagOccurencesPerLattice(h):
    latticesToOccurancesMap = defaultdict(list)
    for l, oc in h['oc']:
        lid = getLatticeLid(l, LATTICE_ACCURACY)
        if lid!='0.0000_0.0000': latticesToOccurancesMap[lid].append(oc)
    return dict([(k,v) for k, v in latticesToOccurancesMap.iteritems() if len(v)>=MIN_HASHTAG_OCCURENCES_PER_LATTICE])

def getSourceLattice(occ):
    occs = occ[:NO_OF_EARLY_LIDS_TO_DETERMINE_SOURCE_LATTICE]
    if occs: return max([(lid, len(list(l))) for lid, l in groupby(sorted([t[0] for t in occs]))], key=lambda t: t[1])
    
def getTimeUnitsAndTimeSeries(occurences, **kwargs):
    occurranceDistributionInEpochs = getOccurranceDistributionInEpochs(occurences, fillInGaps=True, **kwargs)
    if not occurranceDistributionInEpochs:
        print 'x'
    return zip(*sorted(occurranceDistributionInEpochs.iteritems(), key=itemgetter(0)))

def getRadius(locations):
    meanLid = getCenterOfMass(locations,accuracy=LATTICE_ACCURACY)
    distances = [getHaversineDistance(meanLid, p) for p in locations]
    _, upperBoundForDistance = getOutliersRangeUsingIRQ(distances)
    return np.mean(filter(lambda d: d<=upperBoundForDistance, distances))

class HashtagsClassifier:
    PERIODICITY_ID_SLOW_BURST = 'slow_burst'
    PERIODICITY_ID_SUDDEN_BURST = 'sudden_burst'
    LOCALITY_ID_LOCAL = 'local'
    LOCALITY_ID_NON_LOCAL = 'non_local'
    
    RADIUS_LIMIT_FOR_LOCAL_HASHTAG_IN_MILES=500
    PERCENTAGE_OF_OCCURANCES_IN_SUB_ACTIVITY_REGION=1.0
    CLASSIFIER_TIME_UNIT_IN_SECONDS = 5*60
    
    classes = {
               'slow_burst_::_local':0,
               'slow_burst_::_non_local':1,
               'sudden_burst_::_local':2,
               'sudden_burst_::_non_local':3
               }
    
    @staticmethod
    def getId(locality, periodicity): return HashtagsClassifier.classes['%s_::_%s'%(HashtagsClassifier.PERIODICITY_ID_SLOW_BURST, locality)]
#    def getId(locality, periodicity): return HashtagsClassifier.classes['%s_::_%s'%(periodicity, locality)]
    @staticmethod
    def classify(hashtagObject): 
        periodicityId = HashtagsClassifier.getPeriodicityClass(hashtagObject)
        if periodicityId: return HashtagsClassifier.getId(HashtagsClassifier.getHastagLocalityClassForHighestActivityPeriod(hashtagObject), periodicityId)
#        else: return HashtagsClassifier.getId(HashtagsClassifier.getHastagLocalityClassForAllActivityPeriod(hashtagObject), periodicityId)
    @staticmethod
    def getHastagLocalityClassForHighestActivityPeriod(hashtagObject): 
        occuranesInHighestActiveRegion = getOccuranesInHighestActiveRegion(hashtagObject)
        if getRadius(zip(*occuranesInHighestActiveRegion)[0])>=HashtagsClassifier.RADIUS_LIMIT_FOR_LOCAL_HASHTAG_IN_MILES: return HashtagsClassifier.LOCALITY_ID_NON_LOCAL
        else: return HashtagsClassifier.LOCALITY_ID_LOCAL
    @staticmethod
    def getPeriodicityClass(hashtagObject):
        occurranceDistributionInEpochs = getOccurranceDistributionInEpochs(hashtagObject['oc'], timeUnit=HashtagsClassifier.CLASSIFIER_TIME_UNIT_IN_SECONDS)
        if occurranceDistributionInEpochs:
            startEpoch, endEpoch = min(occurranceDistributionInEpochs, key=itemgetter(0))[0], max(occurranceDistributionInEpochs, key=itemgetter(0))[0]
            dataX = range(startEpoch, endEpoch, HashtagsClassifier.CLASSIFIER_TIME_UNIT_IN_SECONDS)
            occurranceDistributionInEpochs = dict(occurranceDistributionInEpochs)
            for x in dataX: 
                if x not in occurranceDistributionInEpochs: occurranceDistributionInEpochs[x]=0
            timeUnits, timeSeries = zip(*sorted(occurranceDistributionInEpochs.iteritems(), key=itemgetter(0)))
            _, _, sizeOfMaxActivityRegion = max(getActiveRegions(timeSeries), key=itemgetter(2))
            activityRegionsWithActivityAboveThreshold=[]
            for start, end, size in getActiveRegions(timeSeries):
                if size>=HashtagsClassifier.PERCENTAGE_OF_OCCURANCES_IN_SUB_ACTIVITY_REGION*sizeOfMaxActivityRegion: activityRegionsWithActivityAboveThreshold.append([start, end, size]) 
            hashtagPropagatingRegion = activityRegionsWithActivityAboveThreshold[0]
            validTimeUnits = [timeUnits[i] for i in range(hashtagPropagatingRegion[0], hashtagPropagatingRegion[1]+1)]
            if timeSeries[list(timeUnits).index(validTimeUnits[0])]>50: return HashtagsClassifier.PERIODICITY_ID_SUDDEN_BURST
            return HashtagsClassifier.PERIODICITY_ID_SLOW_BURST
    stats = {
             0: {'meanTimePeriod': 35.482895038953899, 'outlierBoundary': 124.91611111114999},
             1: {'meanTimePeriod': 43.219448991971397, 'outlierBoundary': 125.008194444525},
             2: {'meanTimePeriod': 24.213369648318182, 'outlierBoundary': 84.581805555495009},
             3: {'meanTimePeriod': 37.167267115601007, 'outlierBoundary': 104.53444444447501}
             }

class MRAreaAnalysis(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(MRAreaAnalysis, self).__init__(*args, **kwargs)
        self.hashtags = defaultdict(list)
    ''' Count number of occurrences, no of hashtags, no of completed hashtags.
    '''
    def mapCountOccurrences(self, key, line):
        for h, d in iterateHashtagObjectInstances(line): yield 'hashtags', h
        yield 'oc', 1
    def reduceCountOccurrences(self, key, values):
        dataToOutput = {}
        if key=='oc': dataToOutput[key] = sum(values)
        elif key=='hashtags': dataToOutput[key] = len(set(values))
        yield 'stats', dataToOutput
    ''' Start: Methods to get hashtag objects
    '''
    def parse_hashtag_objects(self, key, line):
        if False: yield # I'm a generator!
        for h, d in iterateHashtagObjectInstances(line): self.hashtags[h].append(d)
    def parse_hashtag_objects_final(self):
        for h, instances in self.hashtags.iteritems(): # e = earliest, l = latest
            yield h, {'oc': instances, 'e': min(instances, key=lambda t: t[1]), 'l': max(instances, key=lambda t: t[1])}
    def combine_hashtag_instances_without_ending_window(self, key, values):
        hashtagObject = getHashtagWithoutEndingWindow(key, values)
        if hashtagObject: yield key, hashtagObject 
    def combine_hashtag_instances_with_ending_window(self, key, values):
        hashtagObject = getHashtagWithEndingWindow(key, values)
        if hashtagObject: yield key, hashtagObject 
    def add_source_to_hashtag_objects(self, key, hashtagObject):
        occuranesInHighestActiveRegion, isFirstActiveRegion = getOccuranesInHighestActiveRegion(hashtagObject, True)
        if occuranesInHighestActiveRegion:
            lid, count = getSourceLattice(occuranesInHighestActiveRegion)
            if isFirstActiveRegion and count>=MIN_OCCURRENCES_TO_DETERMINE_SOURCE_LATTICE: 
                hashtagObject['source'] = lid
                yield hashtagObject['h'], hashtagObject
    ''' End: Methods to get hashtag objects
    '''
            
    ''' Start: Methods to build lattice graph.
        E(Place_a, Place_b) = len(Hastags(Place_a) and Hastags(Place_b)) / len(Hastags(Place_a))
    '''
    def buildLatticeGraphMap(self, key, hashtagObject):
        hashtagObject['oc']=getOccuranesInHighestActiveRegion(hashtagObject)
        lattices = filterLatticesByMinHashtagOccurencesPerLattice(hashtagObject).keys()
        latticesToOccranceTimeMap = {}
        for k, v in hashtagObject['oc']:
            lid = getLatticeLid(k, LATTICE_ACCURACY)
            if lid!='0.0000_0.0000' and lid in lattices:
                if lid not in latticesToOccranceTimeMap: latticesToOccranceTimeMap[lid]=v
        lattices = latticesToOccranceTimeMap.items()
        if lattices:
            hastagStartTime, hastagEndTime = min(lattices, key=itemgetter(1))[1], max(lattices, key=itemgetter(1))[1]
            hashtagTimePeriod = hastagEndTime - hastagStartTime
            for lattice in lattices: 
                yield lattice[0], ['h', [[hashtagObject['h'], [lattice[1], hashtagTimePeriod]]]]
                yield lattice[0], ['n', lattices]
    def buildLatticeGraphReduce1(self, lattice, values):
        latticeObject = {'h': [], 'n': []}
        for type, value in values: latticeObject[type]+=value
        for k in latticeObject.keys()[:]: latticeObject[k]=dict(latticeObject[k])
        del latticeObject['n'][lattice]
        for k in latticeObject.keys()[:]: latticeObject[k]=latticeObject[k].items()
        neighborLatticeIds = latticeObject['n']; del latticeObject['n']
        if neighborLatticeIds and len(latticeObject['h'])>=MIN_UNIQUE_HASHTAG_OCCURENCES_PER_LATTICE and latticeIdInValidAreas(lattice):
            latticeObject['id'] = lattice
            yield lattice, ['o', latticeObject]
            for no,_ in neighborLatticeIds: yield no, ['no', [lattice, latticeObject['h']]]
    def buildLatticeGraphReduce2(self, lattice, values):
        nodeObject, latticeObject, neighborObjects = {'links':{}, 'id': lattice, 'hashtags': []}, None, []
        for type, value in values:
            if type=='o': latticeObject = value
            else: neighborObjects.append(value)
        if latticeObject:
            currentObjectHashtagsDict = dict(latticeObject['h'])
            currentObjectHashtags = set(currentObjectHashtagsDict.keys())
            nodeObject['hashtags'] = currentObjectHashtagsDict
            for no, neighborHashtags in neighborObjects:
                neighborHashtagsDict=dict(neighborHashtags)
                commonHashtags = currentObjectHashtags.intersection(set(neighborHashtagsDict.keys()))
                if len(commonHashtags)>=MIN_COMMON_HASHTAG_OCCURENCES_BETWEEN_LATTICE_PAIRS: nodeObject['links'][no] = neighborHashtagsDict
            if nodeObject['links']: yield lattice, nodeObject
    ''' End: Methods to build lattice graph..
    '''
    def jobsToCountOccurrences(self): return [(self.mapCountOccurrences, self.reduceCountOccurrences)] 
    def jobsToGetHastagObjectsWithEndingWindow(self): return [self.mr(mapper=self.parse_hashtag_objects, mapper_final=self.parse_hashtag_objects_final, reducer=self.combine_hashtag_instances_with_ending_window)]
    def jobsToGetHastagObjectsWithoutEndingWindow(self): return [self.mr(mapper=self.parse_hashtag_objects, mapper_final=self.parse_hashtag_objects_final, reducer=self.combine_hashtag_instances_without_ending_window)]
    def jobsToGetHastagObjectsWithKnownSource(self): return [self.mr(mapper=self.parse_hashtag_objects, mapper_final=self.parse_hashtag_objects_final, reducer=self.combine_hashtag_instances_without_ending_window)] + \
                                                            [(self.add_source_to_hashtag_objects, None)]
#    def jobsToBuildLatticeGraph(self): return self.jobsToGetHastagObjectsWithoutEndingWindow()+\
    def jobsToBuildLatticeGraph(self): return self.jobsToGetHastagObjectsWithEndingWindow()+\
             [(self.buildLatticeGraphMap, self.buildLatticeGraphReduce1), 
              (self.emptyMapper, self.buildLatticeGraphReduce2)
                ]
    def jobToBuildLocationInAndOutTemporalClosenessGraph(self): return self.jobsToGetHastagObjectsWithoutEndingWindow()+[(self.buildLocationInAndOutTemporalClosenessGraphMap, self.buildLocationInAndOutTemporalClosenessGraphReduce)] 
    

    def steps(self):
        pass
        return self.jobsToCountOccurrences()
#        return self.jobsToGetHastagObjectsWithEndingWindow()
#        return self.jobsToGetHastagObjectsWithoutEndingWindow()
#        return self.jobsToGetHastagObjectsWithKnownSource()
#        return self.jobsToBuildLatticeGraph() 
#        return self.jobToBuildLocationTemporalClosenessGraph()
    
if __name__ == '__main__':
    MRAreaAnalysis.run()
