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

ACCURACY = 0.145
PERCENTAGE_OF_EARLY_LIDS_TO_DETERMINE_SOURCE_LATTICE=0.01

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
                yield key, {'h': key, 't': numberOfInstances, 'e':e, 'l':l, 'oc': occurences}
    def combine_hashtag_instances_without_ending_window(self, key, values):
        occurences = []
        for instances in values: occurences+=instances['oc']
        e, l = min(occurences, key=lambda t: t[1]), max(occurences, key=lambda t: t[1])
        numberOfInstances=len(occurences)
        if numberOfInstances>=MIN_HASHTAG_OCCURENCES and \
            e[1]>=HASHTAG_STARTING_WINDOW:
                yield key, {'h': key, 't': numberOfInstances, 'e':e, 'l':l, 'oc': occurences}
    ''' End: Methods to get hashtag objects
    '''
            
    def addSourceLatticeToHashTagObject(self, key, hashtagObject):
        def getMeanDistanceFromSource(source, llids): return np.mean([getHaversineDistance(source, p) for p in llids])
        sortedOcc = sorted(hashtagObject['oc'], key=lambda t: t[1])[:int(PERCENTAGE_OF_EARLY_LIDS_TO_DETERMINE_SOURCE_LATTICE*len(hashtagObject['oc']))]
#        hashtagObject['src'] = sorted([(lid, len(list(l))) for lid, l in groupby(sorted([t[0] for t in sortedOcc]))], key=lambda t: t[1])[-1]
        llids = sorted([t[0] for t in sortedOcc])
        hashtagObject['src'] = min([(lid, getMeanDistanceFromSource(lid, llids)) for lid in llids], key=lambda t: t[1])
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

    def doHashtagCenterOfMassAnalysis(self,  key, hashtagObject): 
        percentageOfEarlyLattices = [0.01*i for i in range(1, 10)] + [0.1*i for i in range(1, 11)]
        sortedOcc = sorted(hashtagObject['oc'], key=lambda t: t[1])
        llids = [t[0] for t in sortedOcc]; epochs = sortedOcc = [t[1] for t in sortedOcc]
        yield key, {
                    'h':hashtagObject['h'], 't': hashtagObject['t'], 
                    'com': [(p, getCenterOfMass(llids[:int(p*len(llids))], accuracy=ACCURACY, error=True)) for p in percentageOfEarlyLattices],
                    'ep': [(0.0, epochs[0])] + [(p, epochs[int(p*len(epochs))-1]) for p in percentageOfEarlyLattices]
                }
    
    def jobsToGetHastagObjects(self): return [self.mr(mapper=self.parse_hashtag_objects, mapper_final=self.parse_hashtag_objects_final, reducer=self.combine_hashtag_instances)]
    def jobsToGetHastagObjectsWithoutEndingWindow(self): return [self.mr(mapper=self.parse_hashtag_objects, mapper_final=self.parse_hashtag_objects_final, reducer=self.combine_hashtag_instances_without_ending_window)]
    def jobsToAddSourceLatticeToHashTagObject(self): return [(self.addSourceLatticeToHashTagObject, None)]
    def jobsToGetHashtagDistributionInTime(self): return self.jobsToGetHastagObjects() + [(self.getHashtagDistributionInTime, None)]
    def jobsToGetHashtagDistributionInLattice(self): return self.jobsToGetHastagObjects() + [(self.getHashtagDistributionInLattice, None)]
#    def jobsToGetAverageHaversineDistance(self): return self.jobsToGetHastagObjectsWithoutEndingWindow() + [(self.getAverageHaversineDistance, None)]
    def jobsToDoHashtagCenterOfMassAnalysisWithoutEndingWindow(self): return self.jobsToGetHastagObjectsWithoutEndingWindow() + [(self.doHashtagCenterOfMassAnalysis, None)] 
    
    def steps(self):
#        return self.jobsToGetHastagObjects() #+ self.jobsToCountNumberOfKeys()
        return self.jobsToGetHastagObjectsWithoutEndingWindow() + self.jobsToAddSourceLatticeToHashTagObject()
#        return self.jobsToGetHashtagDistributionInTime()
#        return self.jobsToGetHashtagDistributionInLattice()
#        return self.jobsToDoHashtagCenterOfMassAnalysisWithoutEndingWindow()

if __name__ == '__main__':
    MRAnalysis.run()