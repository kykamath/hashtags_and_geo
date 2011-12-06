'''
Created on Nov 19, 2011

@author: kykamath
'''
from library.twitter import getDateTimeObjectFromTweetTimestamp
from library.mrjobwrapper import ModifiedMRJob
from library.geo import getLatticeLid, getLattice, isWithinBoundingBox,\
    getLocationFromLid
import cjson, time, datetime
from collections import defaultdict
from itertools import groupby
import numpy as np
from library.classes import GeneralMethods
from itertools import combinations
from operator import itemgetter

AREA_ACCURACY = 0.145
MIN_HASHTAG_OCCURENCES = 500
HASHTAG_STARTING_WINDOW = time.mktime(datetime.datetime(2011, 2, 25).timetuple())
HASHTAG_ENDING_WINDOW = time.mktime(datetime.datetime(2011, 11, 1).timetuple())
TIME_UNIT_IN_SECONDS = 60*60
MIN_NO_OF_TIME_UNITS_IN_INACTIVE_REGION = 12

MIN_TEMPORAL_CLOSENESS_SCORE_FOR_IN_OUT_LINKS = 0.0
MIN_TEMPORAL_CLOSENESS_SCORE = 0.1
MIN_OBSERVATIONS_GREATER_THAN_MIN_TEMPORAL_CLOSENESS_SCORE = 3

MIN_UNIQUE_HASHTAG_OCCURENCES_PER_LATTICE = 25
MIN_HASHTAG_OCCURENCES_PER_LATTICE = 10
MIN_HASHTAG_SHARING_PROBABILITY = 0.1
PERCENTAGE_OF_EARLY_LIDS_TO_DETERMINE_SOURCE_LATTICE = 0.01

AREAS = {
#         'ny': dict(boundary=[[40.491, -74.356], [41.181, -72.612]]), # New York
#         'north_cal': dict(boundary=[[37.068328,-122.640381], [37.924701,-122.178955]]), # North Cal
#         'austin': dict(boundary=[[30.097613,-97.971954], [30.486551,-97.535248]]),
#         'dallas': dict(boundary=[[32.735307,-96.862335], [32.886507,-96.723633]]),
#         'us': dict(boundary=[[24.527135,-127.792969], [49.61071,-59.765625]]),
        'world': dict(boundary=[[-90,-180], [90, 180]]),  
         }

#CURRENT_AREA = AREAS['ny']
#MIN_HASHTAG_OCCURENCES = CURRENT_AREA['min_hashtag_occurenes']
#BOUNDING_BOX = CURRENT_AREA['boundary']
BOUNDARIES  = [v['boundary'] for v in AREAS.itervalues()]

def iterateHashtagObjectInstances(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    point = getLattice(l, AREA_ACCURACY)
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

def getOccurranceDistributionInEpochs(occ): return [(k[0], len(list(k[1]))) for k in groupby(sorted([GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) for t in zip(*occ)[1]]))]

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
    hashtagPropagatingRegion = max(getActiveRegions(timeSeries), key=itemgetter(2))
    validTimeUnits = [timeUnits[i] for i in range(hashtagPropagatingRegion[0], hashtagPropagatingRegion[1]+1)]
    return [(p,t) for p,t in hashtagObject['oc'] if GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) in validTimeUnits]

def filterLatticesByMinHashtagOccurencesPerLattice(h):
    latticesToOccurancesMap = defaultdict(list)
    for l, oc in h['oc']:latticesToOccurancesMap[getLatticeLid(l, AREA_ACCURACY)].append(oc)
    return dict([(k,v) for k, v in latticesToOccurancesMap.iteritems() if len(v)>=MIN_HASHTAG_OCCURENCES_PER_LATTICE])

def temporalScore(lag, width):
    if lag==0: return 1.0
    return 1-np.log(lag)/np.log(width)

class MRAreaAnalysis(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(MRAreaAnalysis, self).__init__(*args, **kwargs)
        self.hashtags = defaultdict(list)
        
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
    ''' End: Methods to get hashtag objects
    '''
            
    ''' Start: Methods to get hashtag co-occurence probabilities among lattices.
        E(Place_a, Place_b) = len(Hastags(Place_a) and Hastags(Place_b)) / len(Hastags(Place_a))
    '''
    def buildHashtagSharingProbabilityGraphMap(self, key, hashtagObject):
#        lattices = list(set([getLatticeLid(l, accuracy=AREA_ACCURACY) for l in zip(*hashtagObject['oc'])[0]]))
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
        if neighborLatticeIds and len(latticeObject['h'])>=MIN_UNIQUE_HASHTAG_OCCURENCES_PER_LATTICE and latticeIdInValidAreas(lattice):
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
    
    ''' Start: Methods to get hashtag co-occurence probabilities among lattices.
        E(Place_a, Place_b) = len(Hastags(Place_a) and Hastags(Place_b)) / len(Hastags(Place_a))
    '''
    def buildHashtagSharingProbabilityGraphWithTemporalClosenessMap(self, key, hashtagObject):
#        lattices = list(set([getLatticeLid(l, accuracy=AREA_ACCURACY) for l in zip(*hashtagObject['oc'])[0]]))
        hashtagObject['oc']=getOccuranesInHighestActiveRegion(hashtagObject)
        lattices = filterLatticesByMinHashtagOccurencesPerLattice(hashtagObject).keys()
        latticesToOccranceTimeMap = {}
        for k, v in hashtagObject['oc']:
            lid = getLatticeLid(k, AREA_ACCURACY)
            if lid in lattices:
                if lid not in latticesToOccranceTimeMap: latticesToOccranceTimeMap[lid]=v
        lattices = latticesToOccranceTimeMap.items()
        if lattices:
            hastagStartTime, hastagEndTime = min(lattices, key=itemgetter(1))[1], max(lattices, key=itemgetter(1))[1]
            hashtagTimePeriod = hastagEndTime - hastagStartTime
            for lattice in lattices: 
                yield lattice[0], ['h', [[hashtagObject['h'], [lattice[1], hashtagTimePeriod]]]]
                yield lattice[0], ['n', lattices]
    def buildHashtagSharingProbabilityGraphWithTemporalClosenessReduce1(self, lattice, values):
        latticeObject = {'h': [], 'n': []}
        for type, value in values: latticeObject[type]+=value
#        for k in latticeObject.keys()[:]: latticeObject[k]=list(set(latticeObject[k]))
        for k in latticeObject.keys()[:]: latticeObject[k]=dict(latticeObject[k])
#        latticeObject['n'].remove(lattice)
        del latticeObject['n'][lattice]
        for k in latticeObject.keys()[:]: latticeObject[k]=latticeObject[k].items()
        neighborLatticeIds = latticeObject['n']; del latticeObject['n']
        if neighborLatticeIds and len(latticeObject['h'])>=MIN_UNIQUE_HASHTAG_OCCURENCES_PER_LATTICE and latticeIdInValidAreas(lattice):
            latticeObject['id'] = lattice
            yield lattice, ['o', latticeObject]
            for no,_ in neighborLatticeIds: yield no, ['no', [lattice, latticeObject['h']]]
    def buildHashtagSharingProbabilityGraphWithTemporalClosenessReduce2(self, lattice, values):
        nodeObject, latticeObject, neighborObjects = {'links':{}, 'id': lattice, 'hashtags': []}, None, []
        for type, value in values:
            if type=='o': latticeObject = value
            else: neighborObjects.append(value)
        if latticeObject:
            currentObjectHashtagsDict = dict(latticeObject['h'])
#            currentObjectHashtags = set(latticeObject['h'])
            currentObjectHashtags = set(currentObjectHashtagsDict.keys())
            nodeObject['hashtags'] = list(currentObjectHashtags)
            for no, neighborHashtags in neighborObjects:
                neighborHashtagsDict=dict(neighborHashtags)
#                neighborHashtags=set(neighborHashtags)
                neighborHashtags=set(neighborHashtagsDict.keys())
                commonHashtags = currentObjectHashtags.intersection(neighborHashtags)
                prob = len(commonHashtags)/float(len(currentObjectHashtags))
#                if prob>=MIN_HASHTAG_SHARING_PROBABILITY: nodeObject['links'][no] =  [list(neighborHashtags), prob, [np.abs(neighborHashtagsDict[h]-currentObjectHashtagsDict[h]) for h in commonHashtags]]
                if prob>=MIN_HASHTAG_SHARING_PROBABILITY: nodeObject['links'][no] =  [list(neighborHashtags), prob, [(neighborHashtagsDict[h][0],currentObjectHashtagsDict[h][0],currentObjectHashtagsDict[h][1]) for h in commonHashtags]]
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
            lid = getLatticeLid(k, AREA_ACCURACY)
            if lid in validLattices:
                if lid not in latticesToOccranceTimeMap: latticesToOccranceTimeMap[lid]=v
        if latticesToOccranceTimeMap:
            latticesOccranceTimeList = latticesToOccranceTimeMap.items()
            hastagStartTime, hastagEndTime = min(latticesOccranceTimeList, key=itemgetter(1))[1], max(latticesOccranceTimeList, key=itemgetter(1))[1]
            hashtagTimePeriod = hastagEndTime - hastagStartTime
            if hashtagTimePeriod:
                latticesOccranceTimeList = [l for l in latticesOccranceTimeList if latticeIdInValidAreas(l[0])]
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
        occuranesInHighestActiveRegion = [(getLatticeLid(k, AREA_ACCURACY), v) for k, v in occuranesInHighestActiveRegion if getLatticeLid(k, AREA_ACCURACY) in validLattices]
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
                    latticesOccranceTimeList = [(t[0], temporalScore(t[1]-hastagStartTime, hashtagTimePeriod)) for t in latticesOccranceTimeList if t[1]>hastagStartTime and latticeIdInValidAreas(t[0])]
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
    
    def jobsToGetHastagObjectsWithoutEndingWindow(self): return [self.mr(mapper=self.parse_hashtag_objects, mapper_final=self.parse_hashtag_objects_final, reducer=self.combine_hashtag_instances_without_ending_window)]
    def jobsToBuildHashtagSharingProbabilityGraph(self): return self.jobsToGetHastagObjectsWithoutEndingWindow()+\
             [(self.buildHashtagSharingProbabilityGraphMap, self.buildHashtagSharingProbabilityGraphReduce1), 
              (self.emptyMapper, self.buildHashtagSharingProbabilityGraphReduce2)
                ]
    def jobsToBuildHashtagSharingProbabilityGraphWithTemporalCloseness(self): return self.jobsToGetHastagObjectsWithoutEndingWindow()+\
             [(self.buildHashtagSharingProbabilityGraphWithTemporalClosenessMap, self.buildHashtagSharingProbabilityGraphWithTemporalClosenessReduce1), 
              (self.emptyMapper, self.buildHashtagSharingProbabilityGraphWithTemporalClosenessReduce2)
                ]
    def jobToBuildLocationTemporalClosenessGraph(self): return self.jobsToGetHastagObjectsWithoutEndingWindow()+[(self.buildLocationTemporalClosenessGraphMap, self.buildLocationTemporalClosenessGraphReduce)] 
    def jobToBuildLocationInAndOutTemporalClosenessGraph(self): return self.jobsToGetHastagObjectsWithoutEndingWindow()+[(self.buildLocationInAndOutTemporalClosenessGraphMap, self.buildLocationInAndOutTemporalClosenessGraphReduce)] 
    

    def steps(self):
#        return self.jobsToGetHastagObjectsWithoutEndingWindow()
#        return self.jobsToBuildHashtagSharingProbabilityGraph()
        return self.jobsToBuildHashtagSharingProbabilityGraphWithTemporalCloseness() 
#        return self.jobToBuildLocationTemporalClosenessGraph()
    
if __name__ == '__main__':
    MRAreaAnalysis.run()
