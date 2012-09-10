'''
Created on Sept 9, 2012

@author: kykamath
'''
from collections import defaultdict
from datetime import datetime
from library.geo import UTMConverter
from library.mrjobwrapper import ModifiedMRJob
from library.twitter import getDateTimeObjectFromTweetTimestamp
from operator import itemgetter
import cjson
import time


ACCURACY = UTMConverter.accuracy_10KM

# Minimum number of hashtag occurrences
# Used by HashtagsExtractor
MIN_HASHTAG_OCCURRENCES = 1000

# Minimum number of hashtag occurrences at a particular utm id.
# Used by HashtagsByUTMId
MIN_HASHTAG_OCCURRENCES_PER_UTM_ID = 500

# Start time for data analysis
START_TIME = datetime(2011, 3, 1)

# Parameters for the MR Job that will be logged.
HASHTAG_STARTING_WINDOW = time.mktime(START_TIME.timetuple())
PARAMS_DICT = dict(
                   MIN_HASHTAG_OCCURRENCES = MIN_HASHTAG_OCCURRENCES,
                    MIN_HASHTAG_OCCURRENCES_PER_UTM_ID = \
                    MIN_HASHTAG_OCCURRENCES_PER_UTM_ID,
                    HASHTAG_STARTING_WINDOW = HASHTAG_STARTING_WINDOW)

def iterateHashtagObjectInstances(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    for h in data['h']: yield h.lower(), [l, t]

class TweetStats(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(TweetStats, self).__init__(*args, **kwargs)
        self.num_of_tweets = 0
    def map_tweet_to_tweet_stats(self, key, line):
        if False: yield # I'm a generator!
        self.num_of_tweets+=1
    def map_final_tweet_to_tweet_stats(self):
        yield 'total_tweets', self.num_of_tweets
    def red_tuo_tweet_stats_and_values_to_tweet_stats(self, key, values):
        yield 'total_tweets', {'total_tweets': sum(values)} 
    def steps(self):
        return [self.mr(
                    mapper=self.map_tweet_to_tweet_stats,
                    mapper_final=self.map_final_tweet_to_tweet_stats,
                    reducer=self.red_tuo_tweet_stats_and_values_to_tweet_stats)
                ]
        
class HashtagsExtractor(ModifiedMRJob):
    '''
    hashtag_object = {'hashtag' : hashtag,
                      'ltuo_occurrence_time_and_occurrence_utm_id': [],
                      'num_of_occurrences' : 0
                    }
    '''
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self,
                 min_hashtag_occurrences = MIN_HASHTAG_OCCURRENCES,
                 *args,
                 **kwargs):
        super(HashtagsExtractor, self).__init__(*args, **kwargs)
        self.min_hashtag_occurrences = min_hashtag_occurrences
        self.mf_hastag_to_ltuo_occ_time_and_occ_utm_id = defaultdict(list)
    def map_tweet_to_hashtag_object(self, key, line):
        if False: yield # I'm a generator!
        for hashtag, (location, occ_time) in \
                iterateHashtagObjectInstances(line):
            utm_id = UTMConverter.getUTMIdFromLatLong(location[0],
                                                      location[1],
                                                      accuracy=ACCURACY)
            self.mf_hastag_to_ltuo_occ_time_and_occ_utm_id[hashtag]\
                            .append((occ_time, utm_id))
    def map_final_tweet_to_hashtag_object(self):
        for hashtag, ltuo_occ_time_and_occ_utm_id in \
                self.mf_hastag_to_ltuo_occ_time_and_occ_utm_id.iteritems():
            hashtag_object = {'hashtag': hashtag,
                              'ltuo_occ_time_and_occ_utm_id': \
                                    ltuo_occ_time_and_occ_utm_id
                             }
            yield hashtag, hashtag_object
    def reduce_tuo_hashtag_and_hashtag_objects_to_combined_hashtag_object(
                                                  self,
                                                  hashtag,
                                                  hashtag_objects):
        combined_hashtag_object = {'hashtag': hashtag,
                                   'ltuo_occ_time_and_occ_utm_id': []
                                }
        for hashtag_object in hashtag_objects:
            combined_hashtag_object['ltuo_occ_time_and_occ_utm_id']+=\
                    hashtag_object['ltuo_occ_time_and_occ_utm_id']
        combined_hashtag_object['num_of_occurrences'] = \
           len(combined_hashtag_object['ltuo_occ_time_and_occ_utm_id']) 
        e = min(combined_hashtag_object['ltuo_occ_time_and_occ_utm_id'], 
                key=lambda t: t[0])
        if combined_hashtag_object['num_of_occurrences'] >= \
                self.min_hashtag_occurrences and \
                e[0]>=HASHTAG_STARTING_WINDOW:
            combined_hashtag_object['ltuo_occ_time_and_occ_utm_id'] = \
                sorted(combined_hashtag_object['ltuo_occ_time_and_occ_utm_id'],
                       key=itemgetter(0,1))
            yield hashtag, combined_hashtag_object
    def jobs_to_extract_hashtags(self):
        return [self.mr(
        mapper=self.map_tweet_to_hashtag_object,
        mapper_final=self.map_final_tweet_to_hashtag_object,
        reducer=
        self.reduce_tuo_hashtag_and_hashtag_objects_to_combined_hashtag_object)
        ]
    def steps(self):
        return self.jobs_to_extract_hashtags()
        
class HashtagsByUTMId(ModifiedMRJob):
    '''
        utm_object = {'utm_id': utm_id
                      'mf_hashtag_to_count': mf_hashtag_to_count,
                      'total_hashtag_count': total_hashtag_count
                    }
    '''
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(HashtagsByUTMId, self).__init__(*args, **kwargs)
        self.mf_utm_id_to_mf_hashtag_to_count = defaultdict(dict)
        self.mf_utm_id_to_total_hashtag_count = defaultdict(int)
        self.hashtags_extractor = \
            HashtagsExtractor(min_hashtag_occurrences=MIN_HASHTAG_OCCURRENCES)
    def map_hashtag_object_to_utm_object(self, hashtag, hashtag_object):
        if False: yield # I'm a generator!
        for occ_time, occ_utm_id in \
                 hashtag_object['ltuo_occ_time_and_occ_utm_id']:
            if hashtag not in self.mf_utm_id_to_mf_hashtag_to_count[occ_utm_id]:
                self.mf_utm_id_to_mf_hashtag_to_count[occ_utm_id][hashtag] = 0.0
            self.mf_utm_id_to_mf_hashtag_to_count[occ_utm_id][hashtag]+=1
            self.mf_utm_id_to_total_hashtag_count[occ_utm_id]+=1
    def map_final_hashtag_object_to_utm_object(self):
        for utm_id, mf_hashtag_to_count in \
                self.mf_utm_id_to_mf_hashtag_to_count.iteritems():
            yield utm_id, {
                           'mf_hashtag_to_count': mf_hashtag_to_count,
                           'total_hashtag_count': 
                                self.mf_utm_id_to_total_hashtag_count[utm_id]
                        }
    def red_tuo_utm_id_and_utm_objects_to_combined_utm_object(self,
                                                             utm_id,
                                                             utm_objects):
        combined_utm_object = {'utm_id': utm_id,
                               'mf_hashtag_to_count': defaultdict(float),
                               'total_hashtag_count' : 0.0
                               }
        for utm_object in utm_objects:
            if utm_object['mf_hashtag_to_count']:
                mf_hashtag_to_count = utm_object['mf_hashtag_to_count']
                for hashtag, count in mf_hashtag_to_count.iteritems():
                    combined_utm_object['mf_hashtag_to_count'][hashtag]+=count
            if utm_object['total_hashtag_count']:
                combined_utm_object['total_hashtag_count']+=\
                                        utm_object['total_hashtag_count']
        if combined_utm_object['total_hashtag_count'] >= \
                MIN_HASHTAG_OCCURRENCES_PER_UTM_ID:
            yield utm_id, combined_utm_object
    def steps(self):
        return self.hashtags_extractor.jobs_to_extract_hashtags() +\
                [self.mr(
                    mapper=self.map_hashtag_object_to_utm_object,
                    mapper_final=self.map_final_hashtag_object_to_utm_object,
                    reducer=
                    self.red_tuo_utm_id_and_utm_objects_to_combined_utm_object)
                ]

if __name__ == '__main__':
    pass
#    TweetStats.run()
    HashtagsExtractor.run()
#    HashtagsByUTMId.run()
