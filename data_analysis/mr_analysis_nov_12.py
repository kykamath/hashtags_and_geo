'''
Created on Nov 9, 2012

@author: krishnakamath
'''
from collections import defaultdict
from datetime import datetime
from itertools import chain
from library.classes import GeneralMethods
from library.mrjobwrapper import ModifiedMRJob
from library.geo import UTMConverter
from library.twitter import getDateTimeObjectFromTweetTimestamp
from operator import itemgetter
import cjson
import time

LOCATION_ACCURACY = 10**4 # UTM boxes in sq.m
MIN_HASHTAG_OCCURRENCES = 50

MIN_HASHTAG_OCCURRENCES_PER_LOCATION = 5

# Start time for data analysis
START_TIME, END_TIME = datetime(2011, 3, 1), datetime(2012, 9, 30)

# Parameters for the MR Job that will be logged.
HASHTAG_STARTING_WINDOW = time.mktime(START_TIME.timetuple())
HASHTAG_ENDING_WINDOW = time.mktime(END_TIME.timetuple())

PARAMS_DICT = dict(
                   PARAMS_DICT = True,
                   LOCATION_ACCURACY = LOCATION_ACCURACY,
                   MIN_HASHTAG_OCCURRENCES = MIN_HASHTAG_OCCURRENCES,
                   HASHTAG_STARTING_WINDOW = HASHTAG_STARTING_WINDOW,
                   HASHTAG_ENDING_WINDOW = HASHTAG_ENDING_WINDOW,
                   MIN_HASHTAG_OCCURRENCES_PER_LOCATION = MIN_HASHTAG_OCCURRENCES_PER_LOCATION
                )


def iterateHashtagObjectInstances(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    for h in data['h']: yield h.lower(), [l, t]

class DataStats(ModifiedMRJob):
    '''
        {"num_of_unique_hashtags": 27,720,408}
        {"num_of_tweets": 2,020,620,405}
        {"num_of_hashtags": 343,053,584}
    '''
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(DataStats, self).__init__(*args, **kwargs)
        self.num_of_tweets = 0
        self.num_of_hashtags = 0
        self.unique_hashtags = set()
#        self.unique_locations = set()
    def mapper(self, key, line):
        if False: yield # I'm a generator!
        self.num_of_tweets+=1
        for hashtag, (location, occ_time) in iterateHashtagObjectInstances(line):
            self.num_of_hashtags+=1
            self.unique_hashtags.add(hashtag)
    def mapper_final(self):
        yield 'num_of_tweets', self.num_of_tweets
        yield 'num_of_hashtags', self.num_of_hashtags
        yield 'unique_hashtags', list(self.unique_hashtags)
    def reducer(self, key, values):
        if key == 'num_of_tweets': yield 'num_of_tweets', {'num_of_tweets': sum(values)}
        elif key == 'num_of_hashtags': yield 'num_of_hashtags', {'num_of_hashtags': sum(values)}
        elif key == 'unique_hashtags':
            hashtags = list(chain(*values))
            yield 'num_of_unique_hashtags', {'num_of_unique_hashtags': len(set(hashtags))}
            
class HashtagObjects(ModifiedMRJob):
    '''
    hashtag_object = {
                      'hashtag' : hashtag,
                      'ltuo_occ_time_and_occ_location': [],
                    }
    '''
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(HashtagObjects, self).__init__(*args, **kwargs)
        self.mf_hastag_to_ltuo_occ_time_and_occ_location = defaultdict(list)
    def mapper(self, key, line):
        if False: yield # I'm a generator!
        for hashtag, (location, occ_time) in iterateHashtagObjectInstances(line):
            location = UTMConverter.getUTMIdInLatLongFormFromLatLong(
                                                                 location[0], location[1], accuracy=LOCATION_ACCURACY
                                                             )
            self.mf_hastag_to_ltuo_occ_time_and_occ_location[hashtag].append((occ_time, location))
    def mapper_final(self):
        for hashtag, ltuo_occ_time_and_occ_location in self.mf_hastag_to_ltuo_occ_time_and_occ_location.iteritems():
            yield hashtag, {'hashtag': hashtag, 'ltuo_occ_time_and_occ_location': ltuo_occ_time_and_occ_location}
    def _get_combined_hashtag_object(self, hashtag, hashtag_objects):
        combined_hashtag_object = {'hashtag': hashtag, 'ltuo_occ_time_and_occ_location': []}
        for hashtag_object in hashtag_objects:
            combined_hashtag_object['ltuo_occ_time_and_occ_location']+=hashtag_object['ltuo_occ_time_and_occ_location']
        return combined_hashtag_object
    def reducer(self, hashtag, hashtag_objects):
        combined_hashtag_object = self._get_combined_hashtag_object(hashtag, hashtag_objects)
        e = min(combined_hashtag_object['ltuo_occ_time_and_occ_location'], key=lambda t: t[0])
        l = max(combined_hashtag_object['ltuo_occ_time_and_occ_location'], key=lambda t: t[0])
        if len(combined_hashtag_object['ltuo_occ_time_and_occ_location']) >= MIN_HASHTAG_OCCURRENCES and \
                e[0]>=HASHTAG_STARTING_WINDOW and l[0]<=HASHTAG_ENDING_WINDOW:
            combined_hashtag_object['ltuo_occ_time_and_occ_location'] = \
                sorted(combined_hashtag_object['ltuo_occ_time_and_occ_location'], key=itemgetter(0))
            yield hashtag, combined_hashtag_object

class HashtagAndLocationDistribution(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(HashtagAndLocationDistribution, self).__init__(*args, **kwargs)
        self.mf_hashtag_to_occurrence_count = defaultdict(float)
        self.mf_location_to_occurrence_count = defaultdict(float)
    def mapper(self, key, line):
        if False: yield # I'm a generator!
        for hashtag, (location, occ_time) in iterateHashtagObjectInstances(line):
            location = UTMConverter.getUTMIdInLatLongFormFromLatLong(
                                                                 location[0], location[1], accuracy=LOCATION_ACCURACY
                                                             )
            self.mf_hashtag_to_occurrence_count[hashtag]+=1
            self.mf_location_to_occurrence_count[location]+=1
    def mapper_final(self):
        for hashtag, occurrence_count in self.mf_hashtag_to_occurrence_count.iteritems():
            yield hashtag, {'count': occurrence_count, 'type': 'hashtag'}
        for location, occurrence_count in self.mf_location_to_occurrence_count.iteritems():
            yield location, {'count': occurrence_count, 'type': 'location'}
    def reducer(self, key, it_object):
        objects = list(it_object)
        count = sum(map(lambda o: o['count'], objects))
        yield '%s_%s'%(objects[0]['type'], count), 1
    def reducer2(self, key, values):
        key_split = key.split('_')
        yield key, [key_split[0], float(key_split[1]), sum(values)]
    def steps(self):
        return [
                self.mr(self.mapper, self.reducer, self.mapper_final),
                self.mr(reducer = self.reducer2),
                ]

class GetDenseHashtags(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(GetDenseHashtags, self).__init__(*args, **kwargs)
    def mapper(self, key, hashtag_object):
        if 'hashtag' in hashtag_object:
            hashtag_object = cjson.decode(hashtag_object)
            ltuo_occ_time_and_occ_location = hashtag_object.get('ltuo_occ_time_and_occ_location', [])
            ltuo_location_and_items = GeneralMethods.group_items_by(ltuo_occ_time_and_occ_location, key=itemgetter(1))
            ltuo_location_and_items = filter(
                                             lambda (location, items): len(items)>=MIN_HASHTAG_OCCURRENCES_PER_LOCATION,
                                             ltuo_location_and_items
                                             )
            hashtag_object['ltuo_occ_time_and_occ_location'] =\
                                                        list(chain(*map(lambda (_, items): items, ltuo_location_and_items)))
            yield hashtag_object['hashtag'], hashtag_object
    def get_jobs(self): return self.steps()

class DenseHashtagStats(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(DenseHashtagStats, self).__init__(*args, **kwargs)
        self.get_dense_hashtags = GetDenseHashtags()
    def mapper(self, key, hashtag_object):
        yield 'unique_hashtags', 1
        yield 'total_hashtag_tuples', len(hashtag_object['ltuo_occ_time_and_occ_location'])
    def reducer(self, key, values): yield key, sum(values)
    def steps(self): return self.get_dense_hashtags.get_jobs() + [self.mr(mapper=self.mapper, reducer=self.reducer)]
        
if __name__ == '__main__':
#    DataStats.run()
#    HashtagObjects.run()
#    HashtagAndLocationDistribution.run()
#    GetDenseHashtags.run()
    DenseHashtagStats.run()
