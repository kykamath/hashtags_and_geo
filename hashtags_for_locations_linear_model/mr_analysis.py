'''
Created on Sept 9, 2012

@author: kykamath
'''
from library.mrjobwrapper import ModifiedMRJob
from library.twitter import getDateTimeObjectFromTweetTimestamp
import cjson
import time

PARAMS_DICT = dict()

def iterateHashtagObjectInstances(line, all_locations = False):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    for h in data['h']: yield h.lower(), [l, t]

class MRTweetStats(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(MRTweetStats, self).__init__(*args, **kwargs)
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

if __name__ == '__main__':
    pass
    MRTweetStats.run()
