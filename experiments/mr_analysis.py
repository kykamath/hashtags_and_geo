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

ACCURACY = 0.145
PERCENTAGE_OF_EARLY_LIDS_TO_DETERMINE_SOURCE_LATTICE = 0.01
HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS = 24*60*60
K_VALUE_FOR_LOCALITY_INDEX = 0.5

#MIN_HASHTAG_OCCURENCES = 1
#HASHTAG_STARTING_WINDOW = time.mktime(datetime.datetime(2011, 2, 1).timetuple())
#HASHTAG_ENDING_WINDOW = time.mktime(datetime.datetime(2011, 11, 30).timetuple())
#MIN_HASHTAG_OCCURENCES_PER_LATTICE = 4
#MIN_HASHTAG_SHARING_PROBABILITY = 0.1

MIN_HASHTAG_OCCURENCES = 1000
HASHTAG_STARTING_WINDOW = time.mktime(datetime.datetime(2011, 3, 1).timetuple())
HASHTAG_ENDING_WINDOW = time.mktime(datetime.datetime(2011, 10, 31).timetuple())
MIN_HASHTAG_OCCURENCES_PER_LATTICE = 10
MIN_HASHTAG_SHARING_PROBABILITY = 0.1

BOUNDARIES_DICT = dict([
        ('us', (100, [[24.527135,-127.792969], [49.61071,-59.765625]])),
        ('na', (100, [[8.05923,-170.859375], [72.395706,-53.789062]])),
        ('sa', (25, [[-58.447733,-120.585937], [13.239945,-35.15625]])),
        ('eu', (25, [[32.842674,-16.523437], [71.856229,50.625]])),
        ('mea', (25, [[-38.272689,-24.257812], [38.548165,63.984375]])),
        ('ap', (25, [[-46.55886,54.492188], [59.175928,176.835938]]))
])


def iterateHashtagObjectInstances(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    for h in data['h']: yield h.lower(), [getLattice(l, ACCURACY), t]

def getLocationBoundaryId(point):
    for id, (_, boundingBox) in BOUNDARIES_DICT.iteritems():
        if isWithinBoundingBox(point, boundingBox): return id

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
        occurences = []
        for instances in values: occurences+=instances['oc']
        e, l = min(occurences, key=lambda t: t[1]), max(occurences, key=lambda t: t[1])
        numberOfInstances=len(occurences)
        if numberOfInstances>=MIN_HASHTAG_OCCURENCES and \
            e[1]>=HASHTAG_STARTING_WINDOW:
                yield key, {'h': key, 't': numberOfInstances, 'e':e, 'l':l, 'oc': sorted(occurences, key=lambda t: t[1])}
    ''' End: Methods to get hashtag objects
    '''
            
    ''' Start: Methods to get boundary specific stats
    '''
    def mapBoundarySpecificStats(self, key, values):
        occurences = []
        for instances in values: occurences+=instances['oc']
        if min(occurences, key=lambda t: t[1])[1]>=HASHTAG_STARTING_WINDOW:
            for occurence in occurences: 
                bid = getLocationBoundaryId(occurence[0])
                if bid: yield bid+':ilab:'+key, 1
    def reduceBoundarySpecificStats(self, key, values):
        bid, hashTag = key.split(':ilab:')
        noOfHashtags = sum(list(values))
        if noOfHashtags>=4: yield bid, [hashTag, noOfHashtags]
    def combineBoundarySpecificStats(self, bid, hashTags):
        yield bid, {'bid': bid, 'hashTags': list(hashTags)}
    ''' End: Methods to get boundary specific stats
    '''
        
    ''' Start: Methods to get hashtag co-occurence probabilities among lattices.
        E(Place_a, Place_b) = len(Hastags(Place_a) and Hastags(Place_b)) / len(Hastags(Place_a))
    '''
    def buildHashtagSharingProbabilityGraphMap(self, key, hashtagObject):
        lattices = list(set([getLatticeLid(l, accuracy=ACCURACY) for l in zip(*hashtagObject['oc'])[0]]))
        for lattice in lattices: 
            yield lattice, ['h', [hashtagObject['h']]]
            yield lattice, ['n', lattices]
    def buildHashtagSharingProbabilityGraphReduce1(self, lattice, values):
        latticeObject = {'h': [], 'n': []}
        for type, value in values: latticeObject[type]+=value
        for k in latticeObject.keys()[:]: latticeObject[k]=list(set(latticeObject[k]))
        latticeObject['n'].remove(lattice)
        neighborLatticeIds = latticeObject['n']; del latticeObject['n']
        if neighborLatticeIds and len(latticeObject['h'])>=MIN_HASHTAG_OCCURENCES_PER_LATTICE:
            latticeObject['id'] = lattice
            yield lattice, ['o', latticeObject]
            for no in neighborLatticeIds: yield no, ['no', [lattice, latticeObject['h']]]
    def buildHashtagSharingProbabilityGraphReduce2(self, lattice, values):
        nodeObject, latticeObject, neighborObjects = {'links':{}, 'id': lattice}, None, []
        for type, value in values:
            if type=='o': latticeObject = value
            else: neighborObjects.append(value)
        if latticeObject:
            currentObjectHashtags = set(latticeObject['h'])
            for no, neighborHashtags in neighborObjects:
                neighborHashtags=set(neighborHashtags)
                prob = len(currentObjectHashtags.intersection(neighborHashtags))/float(len(currentObjectHashtags))
                if prob>=MIN_HASHTAG_SHARING_PROBABILITY: nodeObject['links'][no] =  prob
            yield lattice, nodeObject
    ''' End: Methods to get hashtag co-occurence probabilities among lattices.
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
    def jobsToGetBoundarySpecificStats(self): return [self.mr(mapper=self.parse_hashtag_objects, mapper_final=self.parse_hashtag_objects_final, reducer=self.mapBoundarySpecificStats),
                                                      self.mr(self.emptyMapper, self.reduceBoundarySpecificStats), 
                                                      self.mr(self.emptyMapper, self.combineBoundarySpecificStats)]
    def jobsToAddSourceLatticeToHashTagObject(self): return [(self.addSourceLatticeToHashTagObject, None)]
    def jobsToGetHashtagDistributionInTime(self): return self.jobsToGetHastagObjects() + [(self.getHashtagDistributionInTime, None)]
    def jobsToGetHashtagDistributionInLattice(self): return self.jobsToGetHastagObjects() + [(self.getHashtagDistributionInLattice, None)]
#    def jobsToGetHastagDisplacementInTime(self, method): return self.jobsToGetHastagObjectsWithoutEndingWindow() + self.jobsToAddSourceLatticeToHashTagObject() + \
#                                                    [(method, None)]
    def jobsToGetHashtagDisplacementStats(self): return self.jobsToGetHastagObjectsWithoutEndingWindow() + [(self.getHashtagDisplacementStats, None)]
#    def jobsToGetAverageHaversineDistance(self): return self.jobsToGetHastagObjectsWithoutEndingWindow() + [(self.getAverageHaversineDistance, None)]
#    def jobsToDoHashtagCenterOfMassAnalysisWithoutEndingWindow(self): return self.jobsToGetHastagObjectsWithoutEndingWindow() + [(self.doHashtagCenterOfMassAnalysis, None)] 
    def jobsToAnalayzeLocalityIndexAtK(self): return self.jobsToGetHashtagWithGuranteedSource() + [(self.analayzeLocalityIndexAtK, None)]
    def jobsToGetHashtagWithGuranteedSource(self): return self.jobsToGetHastagObjectsWithoutEndingWindow() + self.jobsToAddSourceLatticeToHashTagObject() + \
                                                        [(self.getHashtagWithGuranteedSource, None)]
    def jobsToBuildHashtagSharingProbabilityGraph(self): return self.jobsToGetHastagObjectsWithoutEndingWindow()+\
             [(self.buildHashtagSharingProbabilityGraphMap, self.buildHashtagSharingProbabilityGraphReduce1), 
              (self.emptyMapper, self.buildHashtagSharingProbabilityGraphReduce2)
                ]
        
    
    def steps(self):
#        return self.jobsToGetHastagObjects() #+ self.jobsToCountNumberOfKeys()
#        return self.jobsToGetHastagObjectsWithoutEndingWindow() #+ self.jobsToAddSourceLatticeToHashTagObject()
#        return self.jobsToGetBoundarySpecificStats()
#        return self.jobsToGetHashtagDistributionInTime()
#        return self.jobsToGetHashtagDistributionInLattice()
#        return self.jobsToDoHashtagCenterOfMassAnalysisWithoutEndingWindow()
#        return self.jobsToGetHastagDisplacementInTime(method=self.addHashtagSpreadInTime)
#        return self.jobsToGetHastagDisplacementInTime(method=self.addHashtagMeanDistanceInTime)
#        return self.jobsToGetHashtagDisplacementStats()
#        return self.jobsToAnalayzeLocalityIndexAtK()
#        return self.jobsToGetHashtagWithGuranteedSource()
        return self.jobsToBuildHashtagSharingProbabilityGraph()
        
if __name__ == '__main__':
    MRAnalysis.run()
