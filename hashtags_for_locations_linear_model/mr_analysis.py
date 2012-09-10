'''
Created on Sept 9, 2012

@author: kykamath
'''
from collections import defaultdict
from library.geo import UTMConverter
from library.mrjobwrapper import ModifiedMRJob
from library.twitter import getDateTimeObjectFromTweetTimestamp
import cjson
import time


ACCURACY = UTMConverter.accuracy_100KM

# Minimum number of hashtag occurrences at a particular utm id.
# Used by HashtagsByUTMId
MIN_HASHTAG_OCCURRENCES_PER_UTM_ID = 1000

PARAMS_DICT = dict(
                   MIN_HASHTAG_OCCURRENCES_PER_UTM_ID = \
                    MIN_HASHTAG_OCCURRENCES_PER_UTM_ID)

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

class HashtagsByUTMId(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(HashtagsByUTMId, self).__init__(*args, **kwargs)
        self.mf_utm_id_to_mf_hashtag_to_count = defaultdict(dict)
        self.mf_utm_id_to_total_hashtag_count = defaultdict(int)
    def map_tweet_to_utm_object(self, key, line):
        if False: yield # I'm a generator!
        for hashtag, (location, _) in iterateHashtagObjectInstances(line):
            utm_id = UTMConverter.getUTMIdFromLatLong(location[0],
                                                      location[1],
                                                      accuracy=ACCURACY)
            if hashtag not in self.mf_utm_id_to_mf_hashtag_to_count[utm_id]:
                self.mf_utm_id_to_mf_hashtag_to_count[utm_id][hashtag] = 0.0
            self.mf_utm_id_to_mf_hashtag_to_count[utm_id][hashtag]+=1
            self.mf_utm_id_to_total_hashtag_count[utm_id]+=1
    def map_final_tweet_to_utm_object(self):
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
        combined_utm_object = {'mf_hashtag_to_count': defaultdict(float),
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
        return [self.mr(
                    mapper=self.map_tweet_to_utm_object,
                    mapper_final=self.map_final_tweet_to_utm_object,
                    reducer=
                    self.red_tuo_utm_id_and_utm_objects_to_combined_utm_object)
                ]

if __name__ == '__main__':
    pass
#    TweetStats.run()
    HashtagsByUTMId.run()
