'''
Created on Dec 15, 2011

@author: kykamath
'''
from library.mrjobwrapper import ModifiedMRJob
import cjson, time
from library.geo import getLattice, getLatticeLid
from library.twitter import getDateTimeObjectFromTweetTimestamp
from library.classes import GeneralMethods
from collections import defaultdict


LATTICE_ACCURACY = 0.145
MIN_HASHTAG_OCCURENCES = 25
TIME_UNIT_IN_SECONDS = 60*60

def iterateHashtagObjectInstances(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t =  GeneralMethods.approximateEpoch(time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple()), TIME_UNIT_IN_SECONDS)
    point = getLatticeLid(l, LATTICE_ACCURACY)
    if point!='0.0000_0.0000':
        for h in data['h']: yield h.lower(), [point, t]
    
def getHashtagWithoutEndingWindow(key, values):
    occurences = []
    for instances in values: 
        for oc in instances['oc']: occurences.append(oc)
    if occurences:
#        e, l = min(occurences, key=lambda t: t[1]), max(occurences, key=lambda t: t[1])
        numberOfInstances=len(occurences)
        if numberOfInstances>=MIN_HASHTAG_OCCURENCES: 
            return {'h': key, 't': numberOfInstances, 'oc': sorted(occurences, key=lambda t: t[1])}

class MRGraph(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(MRGraph, self).__init__(*args, **kwargs)
        self.hashtags = defaultdict(list)
    def parse_hashtag_objects(self, key, line):
        if False: yield # I'm a generator!
        for h, d in iterateHashtagObjectInstances(line): self.hashtags[h].append(d)
    def parse_hashtag_objects_final(self):
        for h, instances in self.hashtags.iteritems(): # e = earliest, l = latest
            yield h, {'oc': instances, 'e': min(instances, key=lambda t: t[1]), 'l': max(instances, key=lambda t: t[1])}
    def combine_hashtag_instances_without_ending_window(self, key, values):
        hashtagObject = getHashtagWithoutEndingWindow(key, values)
        if hashtagObject: yield key, hashtagObject 
    def jobsToGetHastagObjectsWithEndingWindow(self): return [self.mr(mapper=self.parse_hashtag_objects, mapper_final=self.parse_hashtag_objects_final, reducer=self.combine_hashtag_instances_without_ending_window)]
    def steps(self):
        return self.jobsToGetHastagObjectsWithEndingWindow()
if __name__ == '__main__':
    MRGraph.run()