'''
Created on Sep 28, 2012

@author: krishnakamath
'''
from collections import defaultdict
from datetime import datetime
from library.geo import UTMConverter
from library.mrjobwrapper import ModifiedMRJob
from library.r_helper import R_Helper
from library.twitter import getDateTimeObjectFromTweetTimestamp
from operator import itemgetter
import cjson
import time

ACCURACY = 10**4 # UTM boxes in sq.m

# Minimum number of hashtag occurrences
# Used by HashtagsExtractor
MIN_HASHTAG_OCCURRENCES = 1000

## Minimum number of hashtag occurrences at a particular utm id.
## Used by HashtagsByUTMId
#MIN_HASHTAG_OCCURRENCES_PER_UTM_ID = 1250

# Start time for data analysis
START_TIME, END_TIME = datetime(2011, 3, 1), datetime(2012, 7, 31)

# Parameters for the MR Job that will be logged.
HASHTAG_STARTING_WINDOW = time.mktime(START_TIME.timetuple())
HASHTAG_ENDING_WINDOW = time.mktime(END_TIME.timetuple())
PARAMS_DICT = dict(
                   PARAMS_DICT = True,
                   ACCURACY = ACCURACY,
                   MIN_HASHTAG_OCCURRENCES = MIN_HASHTAG_OCCURRENCES,
                   HASHTAG_STARTING_WINDOW = HASHTAG_STARTING_WINDOW,
                   HASHTAG_ENDING_WINDOW = HASHTAG_ENDING_WINDOW
                   )

def iterateHashtagObjectInstances(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    for h in data['h']: yield h.lower(), [l, t]

class HashtagsExtractor(ModifiedMRJob):
    '''
    hashtag_object = {'hashtag' : hashtag,
                      'ltuo_occ_time_and_occ_utm_id': [],
                      'num_of_occurrences' : 0
                    }
    '''
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self,  min_hashtag_occurrences = MIN_HASHTAG_OCCURRENCES, *args, **kwargs):
        super(HashtagsExtractor, self).__init__(*args, **kwargs)
        self.min_hashtag_occurrences = min_hashtag_occurrences
        self.mf_hastag_to_ltuo_occ_time_and_occ_utm_id = defaultdict(list)
    def map_tweet_to_hashtag_object(self, key, line):
        if False: yield # I'm a generator!
        for hashtag, (location, occ_time) in iterateHashtagObjectInstances(line):
            utm_id = UTMConverter.getUTMIdInLatLongFormFromLatLong( location[0], location[1], accuracy=ACCURACY)
            self.mf_hastag_to_ltuo_occ_time_and_occ_utm_id[hashtag].append((occ_time, utm_id))
    def map_final_tweet_to_hashtag_object(self):
        for hashtag, ltuo_occ_time_and_occ_utm_id in self.mf_hastag_to_ltuo_occ_time_and_occ_utm_id.iteritems():
            hashtag_object = {'hashtag': hashtag, 'ltuo_occ_time_and_occ_utm_id': ltuo_occ_time_and_occ_utm_id}
            yield hashtag, hashtag_object
    def _get_combined_hashtag_object(self, hashtag, hashtag_objects):
        combined_hashtag_object = {'hashtag': hashtag, 'ltuo_occ_time_and_occ_utm_id': []}
        for hashtag_object in hashtag_objects:
            combined_hashtag_object['ltuo_occ_time_and_occ_utm_id']+=hashtag_object['ltuo_occ_time_and_occ_utm_id']
        combined_hashtag_object['num_of_occurrences'] = len(combined_hashtag_object['ltuo_occ_time_and_occ_utm_id']) 
        return combined_hashtag_object
    def red_tuo_hashtag_and_hashtag_objects_to_combined_hashtag_object(self, hashtag, hashtag_objects):
        combined_hashtag_object = self._get_combined_hashtag_object(hashtag, hashtag_objects)
        e = min(combined_hashtag_object['ltuo_occ_time_and_occ_utm_id'], key=lambda t: t[0])
        l = max(combined_hashtag_object['ltuo_occ_time_and_occ_utm_id'], key=lambda t: t[0])
        if combined_hashtag_object['num_of_occurrences'] >= \
                self.min_hashtag_occurrences and \
                e[0]>=HASHTAG_STARTING_WINDOW and l[0]<=HASHTAG_ENDING_WINDOW:
            combined_hashtag_object['ltuo_occ_time_and_occ_utm_id'] = \
                sorted(combined_hashtag_object['ltuo_occ_time_and_occ_utm_id'], key=itemgetter(0))
            yield hashtag, combined_hashtag_object
    def jobs_to_extract_hashtags(self):
        return [self.mr(
                    mapper = self.map_tweet_to_hashtag_object,
                    mapper_final = self.map_final_tweet_to_hashtag_object,
                    reducer = self.red_tuo_hashtag_and_hashtag_objects_to_combined_hashtag_object
                )]
    def steps(self): return self.jobs_to_extract_hashtags()
    
if __name__ == '__main__':
    pass
    HashtagsExtractor.run()
