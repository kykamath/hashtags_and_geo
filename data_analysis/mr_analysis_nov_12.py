'''
Created on Nov 9, 2012

@author: krishnakamath
'''
'''
Created on May 7, 2012

@author: krishnakamath
'''
from itertools import chain
import cjson, time
from library.mrjobwrapper import ModifiedMRJob
from library.twitter import getDateTimeObjectFromTweetTimestamp
#from library.geo import getLatticeLid, getLocationFromLid, getRadiusOfGyration
#from library.classes import GeneralMethods
#from collections import defaultdict
#from datetime import datetime
#from library.stats import entropy, focus
#from operator import itemgetter
#import numpy as np


def iterateHashtagObjectInstances(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    for h in data['h']: yield h.lower(), [l, t]
    
class DataStats(ModifiedMRJob):
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
        if key == 'num_of_tweets': yield 'num_of_tweets', sum(values)
        elif key == 'num_of_hashtags': yield 'num_of_hashtags', sum(values)
        elif key == 'unique_hashtags':
            hashtags = list(chain(*values))
            yield 'num_of_unique_hashtags', len(set(hashtags))
        
if __name__ == '__main__':
    DataStats.run()
