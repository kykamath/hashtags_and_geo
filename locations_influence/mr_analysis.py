'''
Created on Nov 19, 2011

@author: kykamath
'''
from library.twitter import getDateTimeObjectFromTweetTimestamp
from library.mrjobwrapper import ModifiedMRJob
from library.geo import getLatticeLid, getLattice, isWithinBoundingBox,\
    getLocationFromLid
import cjson, time
from datetime import datetime
from collections import defaultdict
from itertools import groupby
from library.classes import GeneralMethods
from operator import itemgetter

# General parameters
LOCATION_ACCURACY = 0.725

# Time windows.
#START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER = datetime(2011, 5, 1), datetime(2011, 12, 31), 'complete_prop' # Complete propagation duration
START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER = datetime(2012, 1, 1), datetime(2012, 3, 16), 'complete_prop' # Complete propagation duration

# Paramters to filter hashtags.
MIN_HASHTAG_OCCURENCES = 750

# Parameters to filter hashtags at a location.
MIN_HASHTAG_OCCURRENCES_AT_A_LOCATION = 0
MIN_NO_OF_UNIQUE_HASHTAGS_AT_A_LOCATION_PER_TIME_UNIT = 0

# Parameters specific to lattice graphs
MIN_COMMON_HASHTAG_OCCURENCES_BETWEEN_LATTICE_PAIRS = 5
MIN_NO_OF_TIME_UNITS_IN_INACTIVE_REGION = 12
MIN_UNIQUE_HASHTAG_OCCURENCES_PER_LATTICE = 5
MIN_HASHTAG_OCCURENCES_PER_LATTICE = 5
BOUNDARIES  = [[[-90,-180], [90, 180]]]

# Time unit.
TIME_UNIT_IN_SECONDS = 60*60


HASHTAG_STARTING_WINDOW, HASHTAG_ENDING_WINDOW = time.mktime(START_TIME.timetuple()), time.mktime(END_TIME.timetuple())

# Parameters for the MR Job that will be logged.
PARAMS_DICT = dict(PARAMS_DICT = True,
                   LOCATION_ACCURACY=LOCATION_ACCURACY,
                   MIN_HASHTAG_OCCURENCES=MIN_HASHTAG_OCCURENCES,
                   HASHTAG_STARTING_WINDOW = HASHTAG_STARTING_WINDOW, HASHTAG_ENDING_WINDOW = HASHTAG_ENDING_WINDOW,
                   MIN_HASHTAG_OCCURRENCES_AT_A_LOCATION = MIN_HASHTAG_OCCURRENCES_AT_A_LOCATION,
                   MIN_NO_OF_UNIQUE_HASHTAGS_AT_A_LOCATION_PER_TIME_UNIT = MIN_NO_OF_UNIQUE_HASHTAGS_AT_A_LOCATION_PER_TIME_UNIT,
                   TIME_UNIT_IN_SECONDS = TIME_UNIT_IN_SECONDS,
                   )


def iterateHashtagObjectInstances(line, all_locations = False):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    point = getLattice(l, LOCATION_ACCURACY)
    if not all_locations:
        pass
#        lattice_lid = getLatticeLid(point, LOCATION_ACCURACY)
#        if lattice_lid in VALID_LOCATIONS_LIST:
#            for h in data['h']: yield h.lower(), [point, t]
    else:
        for h in data['h']: yield h.lower(), [point, t]
        
def getHashtagWithEndingWindow(key, values):
    occurences = []
    for instances in values: 
        for oc in instances['oc']: occurences.append(oc)
    if occurences:
        e, l = min(occurences, key=lambda t: t[1]), max(occurences, key=lambda t: t[1])
        numberOfInstances=len(occurences)
        if numberOfInstances>=MIN_HASHTAG_OCCURENCES and \
            e[1]>=HASHTAG_STARTING_WINDOW and l[1]<=HASHTAG_ENDING_WINDOW: return {'h': key, 't': numberOfInstances, 'e':e, 'l':l, 'oc': sorted(occurences, key=lambda t: t[1])}

class MRAnalysis(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(MRAnalysis, self).__init__(*args, **kwargs)
        self.hashtags = defaultdict(list)
        self.mf_location_to_tuo_hashtag_and_occurrence_time = defaultdict(list)
    ''' Start: Methods to get hashtag objects
    '''
    def mapParseHashtagObjectsForAllLocations(self, key, line):
        if False: yield # I'm a generator!
        for h, d in iterateHashtagObjectInstances(line, all_locations=True): self.hashtags[h].append(d)
    def mapFinalParseHashtagObjects(self):
        for h, instances in self.hashtags.iteritems(): # e = earliest, l = latest
            yield h, {'oc': instances, 'e': min(instances, key=lambda t: t[1]), 'l': max(instances, key=lambda t: t[1])}
    def reduceHashtagInstancesWithEndingWindow(self, key, values):
        hashtagObject = getHashtagWithEndingWindow(key, values)
        if hashtagObject: yield key, hashtagObject
    ''' End: Methods to get hashtag objects
    '''
    ''' Start: Occurrences by location
    '''
    def mapper_hashtag_object_to_tuo_location_and_tuo_hashtag_and_occurrence_time(self, key, hashtag_object):
        if False: yield # I'm a generator!
        for point, t in hashtag_object['oc']:
            location = getLatticeLid(point, LOCATION_ACCURACY)
            self.mf_location_to_tuo_hashtag_and_occurrence_time[location].append([hashtag_object['h'], t])
    def mapper_final_hashtag_object_to_tuo_location_and_tuo_hashtag_and_occurrence_time(self):
        for location, tuo_hashtag_and_occurrence_time in \
                self.mf_location_to_tuo_hashtag_and_occurrence_time.iteritems():
            yield location, tuo_hashtag_and_occurrence_time
    def reducer_tuo_location_and_ito_ltuo_hashtag_and_occurrence_time_to_tuo_location_and_ltuo_hashtag_and_occurrence_time(self, location, ito_ltuo_hashtag_and_occurrence_time):
        ltuo_hashtag_and_occurrence_time = []
        for ino_ltuo_hashtag_and_occurrence_time in ito_ltuo_hashtag_and_occurrence_time:
            ltuo_hashtag_and_occurrence_time+=ino_ltuo_hashtag_and_occurrence_time
        yield location, [location, ltuo_hashtag_and_occurrence_time]
    ''' End: Occurrences by location
    '''
    ''' Start: Methods to build lattice graph.
        E(Place_a, Place_b) = len(Hastags(Place_a) and Hastags(Place_b)) / len(Hastags(Place_a))
    '''
    def buildLatticeGraphMap(self, key, hashtagObject):
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
                lid = getLatticeLid(l, LOCATION_ACCURACY)
                if lid!='0.0000_0.0000': latticesToOccurancesMap[lid].append(oc)
            return dict([(k,v) for k, v in latticesToOccurancesMap.iteritems() if len(v)>=MIN_HASHTAG_OCCURENCES_PER_LATTICE])
        hashtagObject['oc']=getOccuranesInHighestActiveRegion(hashtagObject)
        lattices = filterLatticesByMinHashtagOccurencesPerLattice(hashtagObject).keys()
        latticesToOccranceTimeMap = defaultdict(list)
        for k, v in hashtagObject['oc']:
            lid = getLatticeLid(k, LOCATION_ACCURACY)
            if lid!='0.0000_0.0000' and lid in lattices:
                latticesToOccranceTimeMap[lid].append(v)
        lattices = latticesToOccranceTimeMap.items()
        if lattices:
            hashtagTimePeriod = None
            for lattice in lattices: 
                yield lattice[0], ['h', [[hashtagObject['h'], [lattice[1], hashtagTimePeriod]]]]
                yield lattice[0], ['n', lattices]
    def buildLatticeGraphReduce1(self, lattice, values):
        def latticeIdInValidAreas(latticeId):
            point = getLocationFromLid(latticeId.replace('_', ' '))
            for boundary in BOUNDARIES:
                if isWithinBoundingBox(point, boundary): return True
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
    
    ''' MR Jobs
    '''
    def write_hashtag_objects_file(self):
        return [self.mr(mapper=self.mapParseHashtagObjectsForAllLocations, mapper_final=self.mapFinalParseHashtagObjects, reducer=self.reduceHashtagInstancesWithEndingWindow)] 
    def write_location_objects_file(self): return [self.mr(mapper=self.mapParseHashtagObjectsForAllLocations, mapper_final=self.mapFinalParseHashtagObjects, reducer=self.reduceHashtagInstancesWithEndingWindow)]+\
                 [(self.buildLatticeGraphMap, self.buildLatticeGraphReduce1), 
                  (self.emptyMapper, self.buildLatticeGraphReduce2)
                    ]
    def write_ltuo_location_and_ltuo_hashtag_and_occurrence_time(self):
        return [self.mr(mapper=self.mapParseHashtagObjectsForAllLocations, mapper_final=self.mapFinalParseHashtagObjects, reducer=self.reduceHashtagInstancesWithEndingWindow)] +\
            [self.mr(
                     mapper=self.mapper_hashtag_object_to_tuo_location_and_tuo_hashtag_and_occurrence_time,
                     mapper_final=self.mapper_final_hashtag_object_to_tuo_location_and_tuo_hashtag_and_occurrence_time, 
                     reducer=self.reducer_tuo_location_and_ito_ltuo_hashtag_and_occurrence_time_to_tuo_location_and_ltuo_hashtag_and_occurrence_time
                     )]
    
    
    def steps(self):
        pass
        return self.write_hashtag_objects_file()
#        return self.write_location_objects_file()
#        return self.write_ltuo_location_and_ltuo_hashtag_and_occurrence_time()
if __name__ == '__main__':
    MRAnalysis.run()
