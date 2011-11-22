'''
Created on Nov 19, 2011

@author: kykamath
'''
from library.twitter import getDateTimeObjectFromTweetTimestamp
from library.mrjobwrapper import ModifiedMRJob
from library.geo import getLatticeLid, getHaversineDistance, getLattice,\
    getCenterOfMass, getLocationFromLid
import cjson, time, datetime
from collections import defaultdict
from itertools import groupby
import numpy as np
from library.classes import GeneralMethods

ACCURACY = 0.145
PERCENTAGE_OF_EARLY_LIDS_TO_DETERMINE_SOURCE_LATTICE = 0.01
#HASHTAG_SPREAD_ANALYSIS_WINDOW = datetime.timedelta(seconds=24*4*15*60)
HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS = 60*60

#MIN_HASHTAG_OCCURENCES = 1
#HASHTAG_STARTING_WINDOW = time.mktime(datetime.datetime(2011, 2, 1).timetuple())
#HASHTAG_ENDING_WINDOW = time.mktime(datetime.datetime(2011, 11, 30).timetuple())

MIN_HASHTAG_OCCURENCES = 1000
HASHTAG_STARTING_WINDOW = time.mktime(datetime.datetime(2011, 3, 1).timetuple())
HASHTAG_ENDING_WINDOW = time.mktime(datetime.datetime(2011, 10, 31).timetuple())

def iterateHashtagObjectInstances(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    for h in data['h']: yield h.lower(), [getLattice(l, ACCURACY), t]

def getMeanDistanceFromSource(source, llids): return np.mean([getHaversineDistance(source, p) for p in llids])

#def getMeanDistanceFromSource(source, llids): 
#    for p in llids: 
#        print source, p, getHaversineDistance(source, p)
#    return np.mean([getHaversineDistance(source, p) for p in llids])

def getMeanDistanceBetweenLids(_, llids): 
    meanLid = getCenterOfMass(llids,accuracy=ACCURACY)
    return getMeanDistanceFromSource(meanLid, llids)

def addSourceLatticeToHashTagObject(hashtagObject):
    sortedOcc = hashtagObject['oc'][:int(PERCENTAGE_OF_EARLY_LIDS_TO_DETERMINE_SOURCE_LATTICE*len(hashtagObject['oc']))]
    if len(hashtagObject['oc'])>1000: sortedOcc = hashtagObject['oc'][:10]
    else: sortedOcc = hashtagObject['oc'][:int(PERCENTAGE_OF_EARLY_LIDS_TO_DETERMINE_SOURCE_LATTICE*len(hashtagObject['oc']))]
    llids = sorted([t[0] for t in sortedOcc])
    uniquellids = [getLocationFromLid(l) for l in set(['%s %s'%(l[0], l[1]) for l in llids])]
    sourceLlid = min([(lid, getMeanDistanceFromSource(lid, llids)) for lid in uniquellids], key=lambda t: t[1])
    if sourceLlid[1]>=600: hashtagObject['src'] = max([(lid, len(list(l))) for lid, l in groupby(sorted([t[0] for t in sortedOcc]))], key=lambda t: t[1])
    else: hashtagObject['src'] = sourceLlid

def addHashtagDisplacementsInTime(hashtagObject, distanceMethod=getMeanDistanceFromSource, key='sit'):
#    observedOccurences, currentTime = 0, datetime.datetime.fromtimestamp(hashtagObject['oc'][0][1])
#    spread = []
#    while observedOccurences<len(hashtagObject['oc']):
#        currentTimeWindowBoundary = currentTime+HASHTAG_SPREAD_ANALYSIS_WINDOW
#        llidsToMeasureSpread = [lid for lid, t in hashtagObject['oc'][observedOccurences:] if datetime.datetime.fromtimestamp(t)<currentTimeWindowBoundary]
#        if llidsToMeasureSpread: spread.append([time.mktime(currentTimeWindowBoundary.timetuple()), [len(llidsToMeasureSpread), distanceMethod(hashtagObject['src'][0], llidsToMeasureSpread)]])
#        else: spread.append([time.mktime(currentTimeWindowBoundary.timetuple()), [len(llidsToMeasureSpread), 0]])
#        observedOccurences+=len(llidsToMeasureSpread); currentTime=currentTimeWindowBoundary
#    hashtagObject[key] = spread
    
    spread, occurencesDistribution = [], defaultdict(list)
    for oc in hashtagObject['oc']: occurencesDistribution[GeneralMethods.approximateEpoch(oc[1], HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS)].append(oc)
    for currentTime, oc in occurencesDistribution.iteritems():
        llidsToMeasureSpread = [i[0] for i in oc]
        if llidsToMeasureSpread: spread.append([currentTime, [len(llidsToMeasureSpread), distanceMethod(hashtagObject['src'][0], llidsToMeasureSpread)]])
        else: spread.append([currentTime, [len(llidsToMeasureSpread), 0]])
    hashtagObject[key] = spread

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
            
    def addSourceLatticeToHashTagObject(self, key, hashtagObject):
        addSourceLatticeToHashTagObject(hashtagObject)
        yield key, hashtagObject
    
#    def addHashtagSpreadInTime(self, key, hashtagObject):
#        '''Spread measures the distance from source.
#        '''
#        addHashtagDisplacementsInTime(hashtagObject, distanceMethod=getMeanDistanceFromSource, key='sit')
#        yield key, hashtagObject
#
#    def addHashtagMeanDistanceInTime(self, key, hashtagObject):
#        '''Mean distance measures the mean distance between various occurences of the hastag at that time.
#        '''
#        addHashtagDisplacementsInTime(hashtagObject, distanceMethod=getMeanDistanceBetweenLids, key='mdit')
#        yield key, hashtagObject
        
    def getHashtagDisplacementStats(self, key, hashtagObject):
        ''' Spread measures the distance from source.
            Mean distance measures the mean distance between various occurences of the hastag at that time.
        '''
        addSourceLatticeToHashTagObject(hashtagObject)
        addHashtagDisplacementsInTime(hashtagObject, distanceMethod=getMeanDistanceFromSource, key='sit')
        addHashtagDisplacementsInTime(hashtagObject, distanceMethod=getMeanDistanceBetweenLids, key='mdit')
        yield key, hashtagObject
    
    def getHashtagDistributionInTime(self,  key, hashtagObject):
        distribution = defaultdict(int)
        for _, t in hashtagObject['oc']: distribution[int(t/3600)*3600]+=1
        yield key, {'h':hashtagObject['h'], 't': hashtagObject['t'], 'd': distribution.items()}
    
    def getHashtagDistributionInLattice(self,  key, hashtagObject):
        distribution = defaultdict(int)
        for l, _ in hashtagObject['oc']: distribution[getLatticeLid(l, accuracy=ACCURACY)]+=1
        yield key, {'h':hashtagObject['h'], 't': hashtagObject['t'], 'd': distribution.items()}
    
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
#    def jobsToAddSourceLatticeToHashTagObject(self): return [(self.addSourceLatticeToHashTagObject, None)]
    def jobsToGetHashtagDistributionInTime(self): return self.jobsToGetHastagObjects() + [(self.getHashtagDistributionInTime, None)]
    def jobsToGetHashtagDistributionInLattice(self): return self.jobsToGetHastagObjects() + [(self.getHashtagDistributionInLattice, None)]
#    def jobsToGetHastagDisplacementInTime(self, method): return self.jobsToGetHastagObjectsWithoutEndingWindow() + self.jobsToAddSourceLatticeToHashTagObject() + \
#                                                    [(method, None)]
    def jobsToGetHashtagDisplacementStats(self): return self.jobsToGetHastagObjectsWithoutEndingWindow() + [(self.getHashtagDisplacementStats, None)]
#    def jobsToGetAverageHaversineDistance(self): return self.jobsToGetHastagObjectsWithoutEndingWindow() + [(self.getAverageHaversineDistance, None)]
#    def jobsToDoHashtagCenterOfMassAnalysisWithoutEndingWindow(self): return self.jobsToGetHastagObjectsWithoutEndingWindow() + [(self.doHashtagCenterOfMassAnalysis, None)] 
    
    def steps(self):
#        return self.jobsToGetHastagObjects() #+ self.jobsToCountNumberOfKeys()
#        return self.jobsToGetHastagObjectsWithoutEndingWindow() #+ self.jobsToAddSourceLatticeToHashTagObject()
#        return self.jobsToGetHashtagDistributionInTime()
#        return self.jobsToGetHashtagDistributionInLattice()
#        return self.jobsToDoHashtagCenterOfMassAnalysisWithoutEndingWindow()
#        return self.jobsToGetHastagDisplacementInTime(method=self.addHashtagSpreadInTime)
#        return self.jobsToGetHastagDisplacementInTime(method=self.addHashtagMeanDistanceInTime)
        return self.jobsToGetHashtagDisplacementStats()

if __name__ == '__main__':
    MRAnalysis.run()