'''
Created on Nov 19, 2011

@author: kykamath
'''
from library.twitter import getDateTimeObjectFromTweetTimestamp
from library.mrjobwrapper import ModifiedMRJob
from library.geo import getLatticeLid, getHaversineDistance, getLattice,\
    getCenterOfMass, getLocationFromLid, isWithinBoundingBox
import cjson, time, datetime
from collections import defaultdict
from itertools import groupby
import numpy as np
from library.classes import GeneralMethods
from itertools import combinations
from operator import itemgetter
import math

#ACCURACY = 0.145
PERCENTAGE_OF_EARLY_LIDS_TO_DETERMINE_SOURCE_LATTICE = 0.01
HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS = 24*60*60
K_VALUE_FOR_LOCALITY_INDEX = 0.5
TIME_UNIT_IN_SECONDS = 60*60
MIN_OCCUREANCES_PER_TIME_UNIT = 5
MIN_NO_OF_TIME_UNITS_IN_INACTIVE_REGION = 12
MIN_TEMPORAL_CLOSENESS_SCORE_FOR_IN_OUT_LINKS = 0.0

MIN_HASHTAG_SHARING_PROBABILITY = 0.1
MIN_TEMPORAL_CLOSENESS_SCORE = 0.4
MIN_OBSERVATIONS_GREATER_THAN_MIN_TEMPORAL_CLOSENESS_SCORE = 3

MIN_HASHTAG_OCCURENCES = 1
HASHTAG_STARTING_WINDOW = time.mktime(datetime.datetime(2011, 2, 1).timetuple())
HASHTAG_ENDING_WINDOW = time.mktime(datetime.datetime(2011, 11, 30).timetuple())
MIN_UNIQUE_HASHTAG_OCCURENCES_PER_LATTICE = 4
MIN_HASHTAG_OCCURENCES_PER_LATTICE = 1

#MIN_HASHTAG_OCCURENCES = 500 # Min no. of hashtags observed in the dataset. For example: h1 is valid if it is seen atleast 5 times in the dataset
#HASHTAG_STARTING_WINDOW = time.mktime(datetime.datetime(2011, 2, 25).timetuple())
#HASHTAG_ENDING_WINDOW = time.mktime(datetime.datetime(2011, 11, 1).timetuple())
#MIN_UNIQUE_HASHTAG_OCCURENCES_PER_LATTICE = 25 # Min no. of unique hashtags a lattice should have observed. For example: l1 is valid of it produces [h1, h2, h3] >= 3 (min)
#MIN_HASHTAG_OCCURENCES_PER_LATTICE = 10 # Min no. hashtags lattice should have observed. For example: l1 is valid of it produces [h1, h1, h1] >= 3 (min)

#US_BOUNDARY = ('us', (100, [[24.527135,-127.792969], [49.61071,-59.765625]]))
#CONTINENT_BOUNDARIES_DICT = dict([
#        ('na', (100, [[8.05923,-170.859375], [72.395706,-53.789062]])),
#        ('sa', (25, [[-58.447733,-120.585937], [13.239945,-35.15625]])),
#        ('eu', (25, [[32.842674,-16.523437], [71.856229,50.625]])),
#        ('mea', (25, [[-38.272689,-24.257812], [38.548165,63.984375]])),
#        ('ap', (25, [[-46.55886,54.492188], [59.175928,176.835938]]))
#])

# (Bounding box, MIN_HASHTAG_OCCURENCES)
ACCURACY = 0.0145
AREA_DETAILS = ([[40.491, -74.356], [41.181, -72.612]], 50) # New York

def iterateHashtagObjectInstances(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    for h in data['h']: yield h.lower(), [getLattice(l, ACCURACY), t]

#def getLocationBoundaryId(point):
#    for id, (_, boundingBox) in CONTINENT_BOUNDARIES_DICT.iteritems():
#        if isWithinBoundingBox(point, boundingBox): return id

def filterLatticesByMinHashtagOccurencesPerLattice(h):
    latticesToOccurancesMap = defaultdict(list)
    for l, oc in h['oc']:latticesToOccurancesMap[getLatticeLid(l, ACCURACY)].append(oc)
    return dict([(k,v) for k, v in latticesToOccurancesMap.iteritems() if len(v)>=MIN_HASHTAG_OCCURENCES_PER_LATTICE])

def getMeanDistanceFromSource(source, llids): return np.mean([getHaversineDistance(source, p) for p in llids])

def getLocalityIndexAtK(occurances, kValue):
    ''' Locality index at k - for a hashtag is the minimum radius that covers k percentage of occurrances.
            A high locality index suggests hashtag was global with a small index suggests it was local.
        To find locality index at k, I must find a point that is closest to k percentage of occurances. 
            Brute force requires nC2 complexity. 
            Hence, use lattices of bigger size technique.
    '''
    def getLatticeThatGivesMinimumLocalityIndexAtK():
        occurancesDict = {'occurances': occurances}
        for accuracy in [4, 2, 1, 0.5, ACCURACY]: occurancesDict = getLatticeThatGivesMinimumLocalityIndexAtKForAccuracy(occurancesDict['occurances'], accuracy)
        return occurancesDict['sourceLattice']
    def getLatticeThatGivesMinimumLocalityIndexAtKForAccuracy(occurances, accuracy):
        occurancesDistributionInHigherLattice, distanceMatrix = defaultdict(list), defaultdict(dict)
        for oc in occurances: occurancesDistributionInHigherLattice[getLatticeLid(oc, accuracy)].append(oc)
        higherLattices = sorted(occurancesDistributionInHigherLattice.iteritems(), key=lambda t: len(t[1]), reverse=True)
        for hl1, hl2 in combinations(occurancesDistributionInHigherLattice, 2): distanceMatrix[hl1][hl2] = distanceMatrix[hl2][hl1] = getHaversineDistance(getLocationFromLid(hl1.replace('_', ' ')), getLocationFromLid(hl2.replace('_', ' ')))
        for k,v in distanceMatrix.iteritems(): distanceMatrix[k] = sorted(v.iteritems(), key=itemgetter(1))
        occurancesToReturn = []
        currentHigherLatticeSet, totalOccurances = {'distance': ()}, float(len(occurances))
        for hl, occs  in higherLattices: 
            higherLatticeSet = {'distance': 0, 'observedOccurances': len(occs), 'lattices': [hl], 'sourceLattice': hl}
            while currentHigherLatticeSet['distance']>higherLatticeSet['distance'] and higherLatticeSet['observedOccurances']/totalOccurances<0.5:
                (l, d) = distanceMatrix[hl][0]; 
                distanceMatrix[hl]=distanceMatrix[hl][1:]
                higherLatticeSet['distance']+=d
                higherLatticeSet['lattices'].append(l)
                higherLatticeSet['observedOccurances']+=len(occurancesDistributionInHigherLattice[l])
            if currentHigherLatticeSet==None or currentHigherLatticeSet['distance']>higherLatticeSet['distance']: currentHigherLatticeSet=higherLatticeSet
        for l in currentHigherLatticeSet['lattices']: occurancesToReturn+=occurancesDistributionInHigherLattice[l]
    #    return {'distance': currentHigherLatticeSet['distance'], 'occurances': occurancesToReturn, 'sourceLattice': getLocationFromLid(currentHigherLatticeSet['sourceLattice'].replace('_', ' '))}
        return {'occurances': occurancesToReturn, 'sourceLattice': getLocationFromLid(currentHigherLatticeSet['sourceLattice'].replace('_', ' '))}
    occurancesDistributionInHigherLattice = defaultdict(int)
    for oc in occurances: occurancesDistributionInHigherLattice[getLatticeLid(oc, ACCURACY)]+=1
    totalOccurances, distance, observedOccuraces = float(len(occurances)), 0, 0
    lattice = getLatticeThatGivesMinimumLocalityIndexAtK()
    sortedLatticeObjects = sorted([(getLocationFromLid(k.replace('_', ' ')), getHaversineDistance(lattice, getLocationFromLid(k.replace('_', ' '))), v) for k, v in occurancesDistributionInHigherLattice.iteritems()],
                 key=itemgetter(1))
    for l, d, oc in sortedLatticeObjects:
        distance=d; observedOccuraces+=oc
        if observedOccuraces/totalOccurances>=kValue: break
    return (d, lattice)

def getMeanDistanceBetweenLids(_, llids): 
    meanLid = getCenterOfMass(llids,accuracy=ACCURACY)
    return getMeanDistanceFromSource(meanLid, llids)

def addSourceLatticeToHashTagObject(hashtagObject):
    sortedOcc = hashtagObject['oc'][:int(PERCENTAGE_OF_EARLY_LIDS_TO_DETERMINE_SOURCE_LATTICE*len(hashtagObject['oc']))]
    hashtagObject['src'] = max([(lid, len(list(l))) for lid, l in groupby(sorted([t[0] for t in sortedOcc]))], key=lambda t: t[1])

#def addSourceLatticeToHashTagObject(hashtagObject):
#    sortedOcc = hashtagObject['oc'][:int(PERCENTAGE_OF_EARLY_LIDS_TO_DETERMINE_SOURCE_LATTICE*len(hashtagObject['oc']))]
#    if len(hashtagObject['oc'])>1000: sortedOcc = hashtagObject['oc'][:10]
##    else: sortedOcc = hashtagObject['oc'][:int(PERCENTAGE_OF_EARLY_LIDS_TO_DETERMINE_SOURCE_LATTICE*len(hashtagObject['oc']))]
#    llids = sorted([t[0] for t in sortedOcc])
#    uniquellids = [getLocationFromLid(l) for l in set(['%s %s'%(l[0], l[1]) for l in llids])]
#    sourceLlid = min([(lid, getMeanDistanceFromSource(lid, llids)) for lid in uniquellids], key=lambda t: t[1])
#    if sourceLlid[1]>=600: hashtagObject['src'] = max([(lid, len(list(l))) for lid, l in groupby(sorted([t[0] for t in sortedOcc]))], key=lambda t: t[1])
#    else: hashtagObject['src'] = sourceLlid

def addHashtagDisplacementsInTime(hashtagObject, distanceMethod=getMeanDistanceFromSource, key='sit'):
    spread, occurencesDistribution = [], defaultdict(list)
    for oc in hashtagObject['oc']: occurencesDistribution[GeneralMethods.approximateEpoch(oc[1], HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS)].append(oc)
    for currentTime, oc in occurencesDistribution.iteritems():
        llidsToMeasureSpread = [i[0] for i in oc]
        if llidsToMeasureSpread: spread.append([currentTime, [len(llidsToMeasureSpread), distanceMethod(hashtagObject['src'][0], llidsToMeasureSpread)]])
        else: spread.append([currentTime, [len(llidsToMeasureSpread), 0]])
    hashtagObject[key] = spread

def addHashtagLocalityIndexInTime(hashtagObject):
    liInTime, occurencesDistribution = [], defaultdict(list)
    for oc in hashtagObject['oc']: occurencesDistribution[GeneralMethods.approximateEpoch(oc[1], HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS)].append(oc)
    for currentTime, oc in occurencesDistribution.iteritems(): liInTime.append([currentTime, getLocalityIndexAtK(zip(*oc)[0], K_VALUE_FOR_LOCALITY_INDEX)])
    hashtagObject['liInTime'] = liInTime

def getHashtagWithoutEndingWindow(key, values, specificToArea=False):
    occurences = []
    for instances in values: 
        if not specificToArea: occurences+=instances['oc']
        else:
            MIN_HASHTAG_OCCURENCES = AREA_DETAILS[1]
            for oc in instances['oc']:
                if isWithinBoundingBox(oc[0], AREA_DETAILS[0]): occurences.append(oc)
    if occurences:
        e, l = min(occurences, key=lambda t: t[1]), max(occurences, key=lambda t: t[1])
        numberOfInstances=len(occurences)
        if numberOfInstances>=MIN_HASHTAG_OCCURENCES and \
            e[1]>=HASHTAG_STARTING_WINDOW: return {'h': key, 't': numberOfInstances, 'e':e, 'l':l, 'oc': sorted(occurences, key=lambda t: t[1])}

def getOccurranceDistributionInEpochs(occ): return [(k[0], len(list(k[1]))) for k in groupby(sorted([GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) for t in zip(*occ)[1]]))]
def getOccurencesFilteredByDistributionInTimeUnits(occ): 
    validTimeUnits = [t[0] for t in getOccurranceDistributionInEpochs(occ) if t[1]>=MIN_OCCUREANCES_PER_TIME_UNIT]
    return [(p,t) for p,t in occ if GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) in validTimeUnits]
def getOccuranesInHighestActiveRegion(hashtagObject):
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
    occurranceDistributionInEpochs = getOccurranceDistributionInEpochs(hashtagObject['oc'])
    startEpoch, endEpoch = min(occurranceDistributionInEpochs, key=itemgetter(0))[0], max(occurranceDistributionInEpochs, key=itemgetter(0))[0]
    dataX = range(startEpoch, endEpoch, TIME_UNIT_IN_SECONDS)
    occurranceDistributionInEpochs = dict(occurranceDistributionInEpochs)
    for x in dataX: 
        if x not in occurranceDistributionInEpochs: occurranceDistributionInEpochs[x]=0
    timeUnits, timeSeries = zip(*sorted(occurranceDistributionInEpochs.iteritems(), key=itemgetter(0)))
#    for k, v in zip(timeUnits, timeSeries):
#        print k, v
    hashtagPropagatingRegion = max(getActiveRegions(timeSeries), key=itemgetter(2))
    validTimeUnits = [timeUnits[i] for i in range(hashtagPropagatingRegion[0], hashtagPropagatingRegion[1]+1)]
    return [(p,t) for p,t in hashtagObject['oc'] if GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) in validTimeUnits]

def temporalScore(lag, width):
    if lag==0: return 1.0
    return 1-np.log(lag)/np.log(width)

class MRAnalysis(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(MRAnalysis, self).__init__(*args, **kwargs)
        self.hashtags = defaultdict(list)
        
    ''' Start: Methods to get hashtag objects
    '''
    def parse_hashtag_objects(self, key, line):
        if False: yield # I'm a generator!
        for h, d in iterateHashtagObjectInstances(line): self.hashtags[h].append(d)
    def parse_hashtag_objects_final(self):
        for h, instances in self.hashtags.iteritems(): # e = earliest, l = latest
            yield h, {'oc': instances, 'e': min(instances, key=lambda t: t[1]), 'l': max(instances, key=lambda t: t[1])}
    def combine_hashtag_instances(self, key, values):
        occurences = []
        for instances in values: occurences+=instances['oc']
        e, l = min(occurences, key=lambda t: t[1]), max(occurences, key=lambda t: t[1])
        numberOfInstances=len(occurences)
        if numberOfInstances>=MIN_HASHTAG_OCCURENCES and \
            e[1]>=HASHTAG_STARTING_WINDOW and l[1]<=HASHTAG_ENDING_WINDOW:
                yield key, {'h': key, 't': numberOfInstances, 'e':e, 'l':l, 'oc': sorted(occurences, key=lambda t: t[1])}
    def combine_hashtag_instances_without_ending_window(self, key, values):
        hashtagObject = getHashtagWithoutEndingWindow(key, values)
        if hashtagObject: yield key, hashtagObject 
    def combine_hashtag_instances_without_ending_window_specific_to_an_area(self, key, values):
        hashtagObject = getHashtagWithoutEndingWindow(key, values, specificToArea=True)
        if hashtagObject: yield key, hashtagObject 
    def combine_hashtag_instances_without_ending_window_and_occurences_filtered_by_distribution_in_time_units(self, key, values):
        hashtagObject = getHashtagWithoutEndingWindow(key, values)
        if hashtagObject: 
            hashtagObject['oc'] = getOccurencesFilteredByDistributionInTimeUnits(hashtagObject['oc'])
            yield key, hashtagObject 
    ''' End: Methods to get hashtag objects
    '''
            
#    ''' Start: Methods to get boundary specific stats
#    '''
#    def mapBoundarySpecificStats(self, key, values):
#        occurences = []
#        for instances in values: occurences+=instances['oc']
#        if min(occurences, key=lambda t: t[1])[1]>=HASHTAG_STARTING_WINDOW:
#            for occurence in occurences: 
#                bid = getLocationBoundaryId(occurence[0])
#                if bid: yield bid+':ilab:'+key, 1
#    def reduceBoundarySpecificStats(self, key, values):
#        bid, hashTag = key.split(':ilab:')
#        noOfHashtags = sum(list(values))
#        if noOfHashtags>=4: yield bid, [hashTag, noOfHashtags]
#    def combineBoundarySpecificStats(self, bid, hashTags):
#        yield bid, {'bid': bid, 'hashTags': list(hashTags)}
#    ''' End: Methods to get boundary specific stats
#    '''
        
    ''' Start: Methods to get hashtag co-occurence probabilities among lattices.
        E(Place_a, Place_b) = len(Hastags(Place_a) and Hastags(Place_b)) / len(Hastags(Place_a))
    '''
    def buildHashtagSharingProbabilityGraphMap(self, key, hashtagObject):
#        lattices = list(set([getLatticeLid(l, accuracy=ACCURACY) for l in zip(*hashtagObject['oc'])[0]]))
        hashtagObject['oc']=getOccuranesInHighestActiveRegion(hashtagObject)
        lattices = filterLatticesByMinHashtagOccurencesPerLattice(hashtagObject).keys()
        for lattice in lattices: 
            yield lattice, ['h', [hashtagObject['h']]]
            yield lattice, ['n', lattices]
    def buildHashtagSharingProbabilityGraphReduce1(self, lattice, values):
        latticeObject = {'h': [], 'n': []}
        for type, value in values: latticeObject[type]+=value
        for k in latticeObject.keys()[:]: latticeObject[k]=list(set(latticeObject[k]))
        latticeObject['n'].remove(lattice)
        neighborLatticeIds = latticeObject['n']; del latticeObject['n']
        if neighborLatticeIds and len(latticeObject['h'])>=MIN_UNIQUE_HASHTAG_OCCURENCES_PER_LATTICE:
            latticeObject['id'] = lattice
            yield lattice, ['o', latticeObject]
            for no in neighborLatticeIds: yield no, ['no', [lattice, latticeObject['h']]]
    def buildHashtagSharingProbabilityGraphReduce2(self, lattice, values):
        nodeObject, latticeObject, neighborObjects = {'links':{}, 'id': lattice, 'hashtags': []}, None, []
        for type, value in values:
            if type=='o': latticeObject = value
            else: neighborObjects.append(value)
        if latticeObject:
            currentObjectHashtags = set(latticeObject['h'])
            nodeObject['hashtags'] = list(currentObjectHashtags)
            for no, neighborHashtags in neighborObjects:
                neighborHashtags=set(neighborHashtags)
                prob = len(currentObjectHashtags.intersection(neighborHashtags))/float(len(currentObjectHashtags))
                if prob>=MIN_HASHTAG_SHARING_PROBABILITY: nodeObject['links'][no] =  [list(neighborHashtags), prob]
            yield lattice, nodeObject
    ''' End: Methods to get hashtag co-occurence probabilities among lattices.
    '''
    
    ''' Start: Methods to get temporal closeness among lattices.
    '''
    def buildLocationTemporalClosenessGraphMap(self, key, hashtagObject):
        occuranesInHighestActiveRegion, latticesToOccranceTimeMap = getOccuranesInHighestActiveRegion(hashtagObject), {}
        hashtagObject['oc']=occuranesInHighestActiveRegion
        validLattices = filterLatticesByMinHashtagOccurencesPerLattice(hashtagObject).keys()
        for k, v in occuranesInHighestActiveRegion:
            lid = getLatticeLid(k, ACCURACY)
            if lid in validLattices:
                if lid not in latticesToOccranceTimeMap: latticesToOccranceTimeMap[lid]=v
        if latticesToOccranceTimeMap:
            latticesOccranceTimeList = latticesToOccranceTimeMap.items()
            hastagStartTime, hastagEndTime = min(latticesOccranceTimeList, key=itemgetter(1))[1], max(latticesOccranceTimeList, key=itemgetter(1))[1]
            hashtagTimePeriod = hastagEndTime - hastagStartTime
            if hashtagTimePeriod:
                for l1, l2 in combinations(latticesOccranceTimeList, 2):
                    score = temporalScore(np.abs(l1[1]-l2[1]),hashtagTimePeriod)
                    if score>=MIN_TEMPORAL_CLOSENESS_SCORE:
                        yield l1[0], [hashtagObject['h'], [l2[0], score]]
                        yield l2[0], [hashtagObject['h'], [l1[0], score]]
    def buildLocationTemporalClosenessGraphReduce(self, lattice, values):
        nodeObject, latticesScoreMap, observedHashtags = {'links':{}, 'id': lattice}, defaultdict(list), set()
        for h, (l, v) in values: observedHashtags.add(h), latticesScoreMap[l].append([h, v])
        if len(observedHashtags)>=MIN_UNIQUE_HASHTAG_OCCURENCES_PER_LATTICE:
            for l in latticesScoreMap: 
                if len(latticesScoreMap[l])>=MIN_OBSERVATIONS_GREATER_THAN_MIN_TEMPORAL_CLOSENESS_SCORE: 
                    hashtags, scores = zip(*latticesScoreMap[l])
                    nodeObject['links'][l]= [hashtags, np.mean(scores)]
            if nodeObject['links']:  yield lattice, nodeObject
    ''' End: Methods to get temporal closeness among lattices.
    '''
            
    ''' Start: Methods to get in and out link temporal closeness among lattices.
    '''
    def buildLocationInAndOutTemporalClosenessGraphMap(self, key, hashtagObject):
        def getSourceLattice(occ):
            sortedOcc = occ[:int(PERCENTAGE_OF_EARLY_LIDS_TO_DETERMINE_SOURCE_LATTICE*len(occ))]
            if sortedOcc: return max([(lid, len(list(l))) for lid, l in groupby(sorted([t[0] for t in sortedOcc]))], key=lambda t: t[1])
        occuranesInHighestActiveRegion, latticesToOccranceTimeMap = getOccuranesInHighestActiveRegion(hashtagObject), {}
        validLattices = filterLatticesByMinHashtagOccurencesPerLattice(hashtagObject).keys()
        occuranesInHighestActiveRegion = [(getLatticeLid(k, ACCURACY), v) for k, v in occuranesInHighestActiveRegion if getLatticeLid(k, ACCURACY) in validLattices]
        if occuranesInHighestActiveRegion:
            sourceLattice = getSourceLattice(occuranesInHighestActiveRegion)
            if sourceLattice:
                sourceLattice = sourceLattice[0]
                for lid, v in occuranesInHighestActiveRegion:
                    if lid not in latticesToOccranceTimeMap: latticesToOccranceTimeMap[lid]=v
                latticesOccranceTimeList = latticesToOccranceTimeMap.items()
                hastagStartTime, hastagEndTime = latticesToOccranceTimeMap[sourceLattice], max(latticesOccranceTimeList, key=itemgetter(1))[1]
                hashtagTimePeriod = hastagEndTime - hastagStartTime
                if hashtagTimePeriod:
                    latticesOccranceTimeList = [(t[0], temporalScore(t[1]-hastagStartTime, hashtagTimePeriod)) for t in latticesOccranceTimeList if t[1]>hastagStartTime]
                    for lattice, score in latticesOccranceTimeList:
                        if score>=MIN_TEMPORAL_CLOSENESS_SCORE_FOR_IN_OUT_LINKS:
                            yield sourceLattice, [hashtagObject['h'], 'out_link', [lattice, score]]
                            yield lattice, [hashtagObject['h'], 'in_link', [sourceLattice, score]]
    def buildLocationInAndOutTemporalClosenessGraphReduce(self, lattice, values):
        nodeObject, latticesScoreMap, observedHashtags = {'in_link':{}, 'out_link':{}, 'id': lattice}, {'in_link': defaultdict(list), 'out_link': defaultdict(list)}, set()
        for h, linkType, (l, v) in values: observedHashtags.add(h), latticesScoreMap[linkType][l].append([h,v])
        if len(observedHashtags)>=MIN_UNIQUE_HASHTAG_OCCURENCES_PER_LATTICE:
            for linkType in latticesScoreMap:
                for l in latticesScoreMap[linkType]: 
                    if len(latticesScoreMap[linkType][l])>=MIN_OBSERVATIONS_GREATER_THAN_MIN_TEMPORAL_CLOSENESS_SCORE: 
                        hashtags, scores = zip(*latticesScoreMap[linkType][l])
                        nodeObject[linkType][l]=[hashtags, np.mean(scores)]
            yield lattice, nodeObject
    ''' End: Methods to get in and out link temporal closeness among lattices.
    '''

    def addSourceLatticeToHashTagObject(self, key, hashtagObject):
        addSourceLatticeToHashTagObject(hashtagObject)
        yield key, hashtagObject
    
    def getHashtagWithGuranteedSource(self, key, hashtagObject):
        if hashtagObject['src'][1]/(hashtagObject['t']*0.01)>=0.5: yield key, hashtagObject
        
    def getHashtagDisplacementStats(self, key, hashtagObject):
        ''' Spread measures the distance from source.
            Mean distance measures the mean distance between various occurences of the hastag at that time.
        '''
        addSourceLatticeToHashTagObject(hashtagObject)
        addHashtagDisplacementsInTime(hashtagObject, distanceMethod=getMeanDistanceFromSource, key='sit')
#        addHashtagDisplacementsInTime(hashtagObject, distanceMethod=getMeanDistanceBetweenLids, key='mdit')
        addHashtagLocalityIndexInTime(hashtagObject)
        yield key, hashtagObject
    
    def getHashtagDistributionInTime(self,  key, hashtagObject):
        distribution = defaultdict(int)
        for _, t in hashtagObject['oc']: distribution[int(t/3600)*3600]+=1
        yield key, {'h':hashtagObject['h'], 't': hashtagObject['t'], 'd': distribution.items()}
    
    def getHashtagDistributionInLattice(self,  key, hashtagObject):
        distribution = defaultdict(int)
        for l, _ in hashtagObject['oc']: distribution[getLatticeLid(l, accuracy=ACCURACY)]+=1
        yield key, {'h':hashtagObject['h'], 't': hashtagObject['t'], 'd': distribution.items()}
    
    def analayzeLocalityIndexAtK(self,  key, hashtagObject):
        occurances = zip(*hashtagObject['oc'])[0]
        hashtagObject['liAtVaryingK'] = [(k, getLocalityIndexAtK(occurances, k)) for k in [0.5+0.05*i for i in range(11)]]
        addSourceLatticeToHashTagObject(hashtagObject)
        yield key, hashtagObject
    
#    def getAverageHaversineDistance(self,  key, hashtagObject): 
#        if hashtagObject['t'] >= 1000:
#            percentageOfEarlyLattices = [0.01*i for i in range(1, 11)]
#            def averageHaversineDistance(llids): 
#                if len(llids)>=2: return np.mean(list(getHaversineDistance(getLattice(l1,accuracy=ACCURACY), getLattice(l2,accuracy=ACCURACY)) for l1, l2 in combinations(llids, 2)))
#            llids = sorted([t[0] for t in hashtagObject['oc'] ], key=lambda t: t[1])
#            yield key, {'h':hashtagObject['h'], 't': hashtagObject['t'], 'ahd': [(p, averageHaversineDistance(llids[:int(p*len(llids))])) for p in percentageOfEarlyLattices]}

#    def doHashtagCenterOfMassAnalysis(self,  key, hashtagObject): 
#        percentageOfEarlyLattices = [0.01*i for i in range(1, 10)] + [0.1*i for i in range(1, 11)]
#        sortedOcc = sorted(hashtagObject['oc'], key=lambda t: t[1])
#        llids = [t[0] for t in sortedOcc]; epochs = sortedOcc = [t[1] for t in sortedOcc]
#        yield key, {
#                    'h':hashtagObject['h'], 't': hashtagObject['t'], 
#                    'com': [(p, getCenterOfMass(llids[:int(p*len(llids))], accuracy=ACCURACY, error=True)) for p in percentageOfEarlyLattices],
#                    'ep': [(0.0, epochs[0])] + [(p, epochs[int(p*len(epochs))-1]) for p in percentageOfEarlyLattices]
#                }
    
    def jobsToGetHastagObjects(self): return [self.mr(mapper=self.parse_hashtag_objects, mapper_final=self.parse_hashtag_objects_final, reducer=self.combine_hashtag_instances)]
    def jobsToGetHastagObjectsWithoutEndingWindow(self): return [self.mr(mapper=self.parse_hashtag_objects, mapper_final=self.parse_hashtag_objects_final, reducer=self.combine_hashtag_instances_without_ending_window)]
    def jobsToGetHastagObjectsWithoutEndingWindowAndOcccurencesFilteredByDistributionInTimeUnits(self): return [self.mr(mapper=self.parse_hashtag_objects, mapper_final=self.parse_hashtag_objects_final, reducer=self.combine_hashtag_instances_without_ending_window_and_occurences_filtered_by_distribution_in_time_units)]
    def jobsToGetHastagObjectsWithoutEndingWindowAndSpecificToAnArea(self): return [self.mr(mapper=self.parse_hashtag_objects, mapper_final=self.parse_hashtag_objects_final, reducer=self.combine_hashtag_instances_without_ending_window_specific_to_an_area)]
#    def jobsToGetBoundarySpecificStats(self): return [self.mr(mapper=self.parse_hashtag_objects, mapper_final=self.parse_hashtag_objects_final, reducer=self.mapBoundarySpecificStats),
#                                                      self.mr(self.emptyMapper, self.reduceBoundarySpecificStats), 
#                                                      self.mr(self.emptyMapper, self.combineBoundarySpecificStats)]
    def jobsToAddSourceLatticeToHashTagObject(self): return [(self.addSourceLatticeToHashTagObject, None)]
    def jobsToGetHashtagDistributionInTime(self): return self.jobsToGetHastagObjects() + [(self.getHashtagDistributionInTime, None)]
    def jobsToGetHashtagDistributionInLattice(self): return self.jobsToGetHastagObjects() + [(self.getHashtagDistributionInLattice, None)]
#    def jobsToGetHastagDisplacementInTime(self, method): return self.jobsToGetHastagObjectsWithoutEndingWindow() + self.jobsToAddSourceLatticeToHashTagObject() + \
#                                                    [(method, None)]
#    def jobsToGetHashtagDisplacementStats(self): return self.jobsToGetHastagObjectsWithoutEndingWindow() + [(self.getHashtagDisplacementStats, None)]
#    def jobsToGetAverageHaversineDistance(self): return self.jobsToGetHastagObjectsWithoutEndingWindow() + [(self.getAverageHaversineDistance, None)]
#    def jobsToDoHashtagCenterOfMassAnalysisWithoutEndingWindow(self): return self.jobsToGetHastagObjectsWithoutEndingWindow() + [(self.doHashtagCenterOfMassAnalysis, None)] 
#    def jobsToAnalayzeLocalityIndexAtK(self): return self.jobsToGetHashtagWithGuranteedSource() + [(self.analayzeLocalityIndexAtK, None)]
#    def jobsToGetHashtagWithGuranteedSource(self): return self.jobsToGetHastagObjectsWithoutEndingWindow() + self.jobsToAddSourceLatticeToHashTagObject() + \
#                                                        [(self.getHashtagWithGuranteedSource, None)]
    def jobsToBuildHashtagSharingProbabilityGraph(self): return self.jobsToGetHastagObjectsWithoutEndingWindow()+\
             [(self.buildHashtagSharingProbabilityGraphMap, self.buildHashtagSharingProbabilityGraphReduce1), 
              (self.emptyMapper, self.buildHashtagSharingProbabilityGraphReduce2)
                ]
    def jobToBuildLocationTemporalClosenessGraph(self): return self.jobsToGetHastagObjectsWithoutEndingWindowAndSpecificToAnArea()+[(self.buildLocationTemporalClosenessGraphMap, self.buildLocationTemporalClosenessGraphReduce)] 
    def jobToBuildLocationInAndOutTemporalClosenessGraph(self): return self.jobsToGetHastagObjectsWithoutEndingWindow()+[(self.buildLocationInAndOutTemporalClosenessGraphMap, self.buildLocationInAndOutTemporalClosenessGraphReduce)] 
    
    def steps(self):
        return self.jobsToGetHastagObjects() #+ self.jobsToCountNumberOfKeys()
#        return self.jobsToGetHastagObjectsWithoutEndingWindow() #+ self.jobsToAddSourceLatticeToHashTagObject()
#        return self.jobsToGetHastagObjectsWithoutEndingWindowAndOcccurencesFilteredByDistributionInTimeUnits()
#        return self.jobsToGetBoundarySpecificStats()
#        return self.jobsToGetHashtagDistributionInTime()
#        return self.jobsToGetHashtagDistributionInLattice()
#        return self.jobsToDoHashtagCenterOfMassAnalysisWithoutEndingWindow()
#        return self.jobsToGetHastagDisplacementInTime(method=self.addHashtagSpreadInTime)
#        return self.jobsToGetHastagDisplacementInTime(method=self.addHashtagMeanDistanceInTime)
#        return self.jobsToGetHashtagDisplacementStats()
#        return self.jobsToAnalayzeLocalityIndexAtK()
#        return self.jobsToGetHashtagWithGuranteedSource()
#        return self.jobsToBuildHashtagSharingProbabilityGraph()
#        return self.jobToBuildLocationTemporalClosenessGraph()
#        return self.jobToBuildLocationInAndOutTemporalClosenessGraph()

#    def steps(self):
#        return self.jobsToGetHastagObjectsWithoutEndingWindowAndSpecificToAnArea()
#        return self.jobsToBuildHashtagSharingProbabilityGraph()
#        return self.jobToBuildLocationTemporalClosenessGraph()
    
if __name__ == '__main__':
    MRAnalysis.run()
