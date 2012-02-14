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

#Local run parameters
#MIN_HASHTAG_OCCURENCES = 1
#START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER = datetime(2011, 1, 1), datetime(2012, 1, 31), 'complete' # Complete duration

# General parameters
LOCATION_ACCURACY = 0.145

# Paramters to filter hashtags.
MIN_HASHTAG_OCCURENCES = 100

# Time windows.
#START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER = datetime(2011, 4, 1), datetime(2012, 1, 31), 'complete' # Complete duration
#START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER = datetime(2011, 5, 1), datetime(2011, 12, 31), 'complete_prop' # Complete propagation duration
#START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER = datetime(2011, 5, 1), datetime(2011, 10, 31), 'training' # Training duration
START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER = datetime(2011, 11, 1), datetime(2011, 12, 31), 'testing' # Testing duration

HASHTAG_STARTING_WINDOW, HASHTAG_ENDING_WINDOW = time.mktime(START_TIME.timetuple()), time.mktime(END_TIME.timetuple())

# Parameters to filter hashtags at a location.
MIN_HASHTAG_OCCURRENCES_AT_A_LOCATION = 3

# Time unit.
TIME_UNIT_IN_SECONDS = 30*60


# Parameters for the MR Job that will be logged.
PARAMS_DICT = dict(PARAMS_DICT = True,
                   LOCATION_ACCURACY=LOCATION_ACCURACY,
                   MIN_HASHTAG_OCCURENCES=MIN_HASHTAG_OCCURENCES,
                   HASHTAG_STARTING_WINDOW = HASHTAG_STARTING_WINDOW, HASHTAG_ENDING_WINDOW = HASHTAG_ENDING_WINDOW,
                   MIN_HASHTAG_OCCURRENCES_AT_A_LOCATION = MIN_HASHTAG_OCCURRENCES_AT_A_LOCATION,
                   TIME_UNIT_IN_SECONDS = TIME_UNIT_IN_SECONDS,
                   )

def iterateHashtagObjectInstances(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    point = getLattice(l, LOCATION_ACCURACY)
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
            
def getLocationObjectForLocationUnits(key, values):
    locationObject = {'loc': key, 'oc': []}
    hashtagObjects = defaultdict(list)
    for instances in values: 
        for h, t in instances['oc']: hashtagObjects[h].append(t)
    hashtagObjects = dict(filter(lambda (j, occs): len(occs)>=MIN_HASHTAG_OCCURRENCES_AT_A_LOCATION, hashtagObjects.iteritems()))
    for h, occs in hashtagObjects.iteritems():
        for oc in occs: locationObject['oc'].append([h, oc])
    if locationObject['oc']: return locationObject

def getTimeUnitObjectFromTimeUnits(key, values):
    timeUnitObject = {'tu': key, 'oc': []}
    for instance in values:  timeUnitObject['oc']+=instance['oc']
    if timeUnitObject['oc']: return timeUnitObject
    

class MRAnalysis(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(MRAnalysis, self).__init__(*args, **kwargs)
        self.hashtags = defaultdict(list)
        self.locations = defaultdict(list)
        self.timeUnits = defaultdict(list)
    ''' Start: Methods to get hashtag objects
    '''
    def mapParseHashtagObjects(self, key, line):
        if False: yield # I'm a generator!
        for h, d in iterateHashtagObjectInstances(line): self.hashtags[h].append(d)
    def mapFinalParseHashtagObjects(self):
        for h, instances in self.hashtags.iteritems(): # e = earliest, l = latest
            yield h, {'oc': instances, 'e': min(instances, key=lambda t: t[1]), 'l': max(instances, key=lambda t: t[1])}
    def reduceHashtagInstancesWithoutEndingWindow(self, key, values):
        hashtagObject = getHashtagWithoutEndingWindow(key, values)
        if hashtagObject: yield key, hashtagObject 
    def reduceHashtagInstancesWithEndingWindow(self, key, values):
        hashtagObject = getHashtagWithEndingWindow(key, values)
        if hashtagObject: yield key, hashtagObject 
    ''' End: Methods to get hashtag objects
    '''
    ''' Start: Methods to get location objects.
    '''
    def mapHashtagObjectsToLocationUnits(self, key, hashtagObject):
        if False: yield # I'm a generator!
        hashtag = hashtagObject['h']
        for point, t in hashtagObject['oc']: 
            self.locations[getLatticeLid(point, LOCATION_ACCURACY)].append([hashtagObject['h'], t])
    def mapFinalHashtagObjectsToLocationUnits(self):
        for loc, occurrences in self.locations.iteritems(): yield loc, {'loc': loc, 'oc': occurrences}
    def reduceLocationUnitsToLocationObject(self, key, values):
        locationObject = getLocationObjectForLocationUnits(key, values)
        if locationObject: yield key, locationObject
    ''' End: Methods to get location objects.
    '''
    ''' Start: Methods to occurrences by time unit.
    '''
    def mapLocationsObjectsToTimeUnits(self, key, locationObject):
        if False: yield # I'm a generator!
        for h, t in locationObject['oc']: self.timeUnits[GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS)].append([h, locationObject['loc'], t])
    def mapFinalLocationsObjectsToTimeUnits(self):
        for t, data in self.timeUnits.iteritems(): yield t, {'tu':t, 'oc': data}
    def reduceTimeUnitsToTimeUnitObject(self, key, values):
        timeUnitObject = getTimeUnitObjectFromTimeUnits(key, values)
        if timeUnitObject: yield key, timeUnitObject
    ''' End: Methods to occurrences by time unit.
    '''
    ''' MR Jobs
    '''
    def jobsToGetHastagObjectsWithEndingWindow(self): return [self.mr(mapper=self.mapParseHashtagObjects, mapper_final=self.mapFinalParseHashtagObjects, reducer=self.reduceHashtagInstancesWithEndingWindow)]
    def jobsToGetHastagObjectsWithoutEndingWindow(self): return [self.mr(mapper=self.mapParseHashtagObjects, mapper_final=self.mapFinalParseHashtagObjects, reducer=self.reduceHashtagInstancesWithoutEndingWindow)]
    def jobsToGetLocationObjects(self): return self.jobsToGetHastagObjectsWithEndingWindow() + [self.mr(mapper=self.mapHashtagObjectsToLocationUnits, mapper_final=self.mapFinalHashtagObjectsToLocationUnits, reducer=self.reduceLocationUnitsToLocationObject)]
    def jobsToGetTimeUnitObjects(self): return self.jobsToGetLocationObjects() + \
                                                [self.mr(mapper=self.mapLocationsObjectsToTimeUnits, mapper_final=self.mapFinalLocationsObjectsToTimeUnits, reducer=self.reduceTimeUnitsToTimeUnitObject)]

    def steps(self):
        pass
#        return self.jobsToGetHastagObjectsWithEndingWindow()
#        return self.jobsToGetHastagObjectsWithoutEndingWindow()
#        return self.jobsToGetLocationObjects()
        return self.jobsToGetTimeUnitObjects()
if __name__ == '__main__':
    MRAnalysis.run()
