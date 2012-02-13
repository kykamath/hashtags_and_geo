'''
Created on Nov 19, 2011

@author: kykamath
'''
from library.twitter import getDateTimeObjectFromTweetTimestamp
from library.mrjobwrapper import ModifiedMRJob
from library.geo import getLatticeLid, getLattice, isWithinBoundingBox,\
    getLocationFromLid, getHaversineDistance, getCenterOfMass
import cjson, time
from datetime import datetime
from collections import defaultdict
from itertools import groupby
import numpy as np
from library.classes import GeneralMethods
from itertools import combinations
from operator import itemgetter
from library.stats import getOutliersRangeUsingIRQ

# General parameters
LATTICE_ACCURACY = 0.145

# Paramters to filter hashtags.
MIN_HASHTAG_OCCURENCES = 250

# Time windows.
startTime, endTime = datetime(2011, 4, 1), datetime(2012, 1, 31) # Complete duration
#startTime, endTime = datetime(2011, 5, 1), datetime(2011, 12, 31) # Complete propagation duration
#startTime, endTime = datetime(2011, 5, 1), datetime(2011, 10, 31) # Training duration
#startTime, endTime = datetime(2011, 11, 1), datetime(2011, 12, 31) # Testing duration
HASHTAG_STARTING_WINDOW, HASHTAG_ENDING_WINDOW = time.mktime(startTime.timetuple()), time.mktime(endTime.timetuple())

# Parameters for the MR Job that will be logged.
PARAMS_DICT = dict(PARAMS_DICT = True,
                   LATTICE_ACCURACY=LATTICE_ACCURACY,
                   MIN_HASHTAG_OCCURENCES=MIN_HASHTAG_OCCURENCES,
                   HASHTAG_STARTING_WINDOW = HASHTAG_STARTING_WINDOW, HASHTAG_ENDING_WINDOW = HASHTAG_ENDING_WINDOW,
                   )

def iterateHashtagObjectInstances(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    point = getLattice(l, LATTICE_ACCURACY)
    for h in data['h']: yield h.lower(), [point, t]

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
    def combine_hashtag_instances_without_ending_window(self, key, values):
        hashtagObject = getHashtagWithoutEndingWindow(key, values)
        if hashtagObject: yield key, hashtagObject 
    def combine_hashtag_instances_with_ending_window(self, key, values):
        hashtagObject = getHashtagWithEndingWindow(key, values)
        if hashtagObject: yield key, hashtagObject 
    ''' End: Methods to get hashtag objects
    '''
    def jobsToGetHastagObjectsWithEndingWindow(self): return [self.mr(mapper=self.parse_hashtag_objects, mapper_final=self.parse_hashtag_objects_final, reducer=self.combine_hashtag_instances_with_ending_window)]
    def jobsToGetHastagObjectsWithoutEndingWindow(self): return [self.mr(mapper=self.parse_hashtag_objects, mapper_final=self.parse_hashtag_objects_final, reducer=self.combine_hashtag_instances_without_ending_window)]
    

    def steps(self):
        pass
        return self.jobsToGetHastagObjectsWithEndingWindow()
#        return self.jobsToGetHastagObjectsWithoutEndingWindow()
    
if __name__ == '__main__':
    MRAnalysis.run()
