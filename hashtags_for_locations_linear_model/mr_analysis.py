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


ACCURACY = 10**5 # UTM boxes in sq.m

ACCURACIES = [10**3, 10**4, 10**5]

# Minimum number of hashtag occurrences
# Used by HashtagsExtractor
MIN_HASHTAG_OCCURRENCES = 750

# Minimum number of hashtag occurrences at a particular utm id.
# Used by HashtagsByUTMId
MIN_HASHTAG_OCCURRENCES_PER_UTM_ID = 500

# Generate utm object with neigbor information
UTM_OBJECT_WITH_NEIGHBOR_INFO = True
UTM_OBJECT_WITH_MIN_COMMON_HASHTAGS= 25

# Start time for data analysis
START_TIME, END_TIME = datetime(2011, 3, 1), datetime(2012, 7, 31)

# Parameters for the MR Job that will be logged.
HASHTAG_STARTING_WINDOW = time.mktime(START_TIME.timetuple())
HASHTAG_ENDING_WINDOW = time.mktime(END_TIME.timetuple())
PARAMS_DICT = dict(PARAMS_DICT = True,
                   MIN_HASHTAG_OCCURRENCES = MIN_HASHTAG_OCCURRENCES,
                   MIN_HASHTAG_OCCURRENCES_PER_UTM_ID = \
                   MIN_HASHTAG_OCCURRENCES_PER_UTM_ID,
                   HASHTAG_STARTING_WINDOW = HASHTAG_STARTING_WINDOW,
                   HASHTAG_ENDING_WINDOW = HASHTAG_ENDING_WINDOW,
                   UTM_OBJECT_WITH_NEIGHBOR_INFO = UTM_OBJECT_WITH_NEIGHBOR_INFO,
                   UTM_OBJECT_WITH_MIN_COMMON_HASHTAGS= UTM_OBJECT_WITH_MIN_COMMON_HASHTAGS)

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
                      'ltuo_occ_time_and_occ_utm_id': [],
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
            utm_id = UTMConverter.getUTMIdInLatLongFormFromLatLong(
                                                    location[0],
                                                    location[1],
                                                    accuracy=ACCURACY)
            self.mf_hastag_to_ltuo_occ_time_and_occ_utm_id[hashtag]\
                            .append((occ_time, utm_id))
    def map_tweet_to_hashtag_object_at_varying_accuracies(self, key, line):
        if False: yield # I'm a generator!
        for hashtag, (location, occ_time) in \
                iterateHashtagObjectInstances(line):
            for accuracy in ACCURACIES:
                utm_id = UTMConverter.getUTMIdInLatLongFormFromLatLong(
                                                        location[0],
                                                        location[1],
                                                        accuracy=accuracy)
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
    def _get_combined_hashtag_object(self, hashtag, hashtag_objects):
        combined_hashtag_object = {'hashtag': hashtag,
                                   'ltuo_occ_time_and_occ_utm_id': []
                                }
        for hashtag_object in hashtag_objects:
            combined_hashtag_object['ltuo_occ_time_and_occ_utm_id']+=\
                    hashtag_object['ltuo_occ_time_and_occ_utm_id']
        combined_hashtag_object['num_of_occurrences'] = \
           len(combined_hashtag_object['ltuo_occ_time_and_occ_utm_id']) 
        return combined_hashtag_object
    def red_tuo_hashtag_and_hashtag_objects_to_combined_hashtag_object(
                                                  self,
                                                  hashtag,
                                                  hashtag_objects):
        combined_hashtag_object = self._get_combined_hashtag_object(
                                                            hashtag,
                                                            hashtag_objects)
        e = min(combined_hashtag_object['ltuo_occ_time_and_occ_utm_id'], 
                key=lambda t: t[0])
        l = max(combined_hashtag_object['ltuo_occ_time_and_occ_utm_id'], 
                key=lambda t: t[0])
        if combined_hashtag_object['num_of_occurrences'] >= \
                self.min_hashtag_occurrences and \
                e[0]>=HASHTAG_STARTING_WINDOW and \
                l[0]<=HASHTAG_ENDING_WINDOW:
            combined_hashtag_object['ltuo_occ_time_and_occ_utm_id'] = \
                sorted(combined_hashtag_object['ltuo_occ_time_and_occ_utm_id'],
                       key=itemgetter(0))
            yield hashtag, combined_hashtag_object
    def red_tuo_hashtag_and_hashtag_objects_to_combined_hashtag_object_at_varying_accuracies(
                                                              self,
                                                              hashtag,
                                                              hashtag_objects):
        combined_hashtag_object = self._get_combined_hashtag_object(
                                                            hashtag,
                                                            hashtag_objects)
        e = min(combined_hashtag_object['ltuo_occ_time_and_occ_utm_id'], 
                key=lambda t: t[0])
        l = max(combined_hashtag_object['ltuo_occ_time_and_occ_utm_id'], 
                key=lambda t: t[0])
        if combined_hashtag_object['num_of_occurrences']/len(ACCURACIES) >= \
                self.min_hashtag_occurrences and \
                e[0]>=HASHTAG_STARTING_WINDOW and \
                l[0]<=HASHTAG_ENDING_WINDOW:
            combined_hashtag_object['ltuo_occ_time_and_occ_utm_id'] = \
                sorted(combined_hashtag_object['ltuo_occ_time_and_occ_utm_id'],
                       key=itemgetter(0))
            yield hashtag, combined_hashtag_object
    def jobs_to_extract_hashtags(self):
        return [self.mr(
        mapper=self.map_tweet_to_hashtag_object,
        mapper_final=self.map_final_tweet_to_hashtag_object,
        reducer=
        self.red_tuo_hashtag_and_hashtag_objects_to_combined_hashtag_object)
        ]
    def jobs_to_extract_hashtags_at_varying_accuracies(self):
        return [self.mr(
        mapper=self.map_tweet_to_hashtag_object_at_varying_accuracies,
        mapper_final=self.map_final_tweet_to_hashtag_object,
        reducer=
        self.red_tuo_hashtag_and_hashtag_objects_to_combined_hashtag_object_at_varying_accuracies)
        ]
    def steps(self):
        return self.jobs_to_extract_hashtags()

class HashtagsDistributionInUTM(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(HashtagsDistributionInUTM, self).__init__(*args, **kwargs)
        self.hashtags_extractor = HashtagsExtractor()
        self.mf_utm_id_to_hashtag_count = defaultdict(int)
    def map_hashtag_object_to_dist_in_utm(self, hashtag, hashtag_object):
        if False: yield # I'm a generator!
        for occurrence_time, utm_id in \
                hashtag_object['ltuo_occ_time_and_occ_utm_id']:
            self.mf_utm_id_to_hashtag_count[utm_id]+=1
    def map_final_hashtag_object_to_dist_in_utm(self):  
        for utm_id, hashtag_count in \
                self.mf_utm_id_to_hashtag_count.iteritems():
            yield utm_id, hashtag_count
    def red_tuo_utm_id_and_hashtag_counts_to_accuracy_and_hashtag_dist(
                                                             self,
                                                             utm_id,
                                                             hashtag_counts):
        hashtags_dist = sum(hashtag_counts)
        if hashtags_dist >= MIN_HASHTAG_OCCURRENCES_PER_UTM_ID:
            accuracy = \
                UTMConverter.getAccuracyFromUTMIdInLatLongFormFrom(utm_id)
            yield accuracy, hashtags_dist
    def red_tuo_accuracy_and_hashtag_dists_to_accuracy_and_dist(self,
                                                                accuracy,
                                                                hashtag_counts):
        yield accuracy, {'accuracy': accuracy,
                         'hashtag_distribution': list(hashtag_counts)}
    def jobs_to_get_utm_accuracy_distribution(self):
        return self.hashtags_extractor.\
                        jobs_to_extract_hashtags_at_varying_accuracies() +\
                [self.mr(
                    mapper=self.map_hashtag_object_to_dist_in_utm,
                    mapper_final=self.map_final_hashtag_object_to_dist_in_utm,
                    reducer=
                    self.red_tuo_utm_id_and_hashtag_counts_to_accuracy_and_hashtag_dist)
                ] +\
                [self.mr(
                    mapper=self.emptyMapper,
                    reducer=
                    self.red_tuo_accuracy_and_hashtag_dists_to_accuracy_and_dist
                    )
                 ]
    def steps(self):
        return self.jobs_to_get_utm_accuracy_distribution()
        
class HashtagsByUTMId(ModifiedMRJob):
    '''
        utm_object = {'utm_id': utm_id
                      'mf_hashtag_to_count': mf_hashtag_to_count,
                      'total_hashtag_count': total_hashtag_count,
                      'mf_nei_utm_id_to_common_h_count':
                                                mf_nei_utm_id_to_common_h_count
                    }
    '''
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(HashtagsByUTMId, self).__init__(*args, **kwargs)
        self.mf_utm_id_to_mf_hashtag_to_count = defaultdict(dict)
        self.mf_utm_id_to_total_hashtag_count = defaultdict(int)
        self.hashtags_extractor = HashtagsExtractor()
        self.mf_utm_id_to_mf_nei_utm_id_and_common_h_count = defaultdict(dict)
    def map_hashtag_object_to_utm_object(self, hashtag, hashtag_object):
        if False: yield # I'm a generator!
        so_utm_ids = set()
        for occ_time, occ_utm_id in \
                 hashtag_object['ltuo_occ_time_and_occ_utm_id']:
            if hashtag not in self.mf_utm_id_to_mf_hashtag_to_count[occ_utm_id]:
                self.mf_utm_id_to_mf_hashtag_to_count[occ_utm_id][hashtag] = 0.0
            self.mf_utm_id_to_mf_hashtag_to_count[occ_utm_id][hashtag]+=1
            self.mf_utm_id_to_total_hashtag_count[occ_utm_id]+=1
            so_utm_ids.add(occ_utm_id)
        if UTM_OBJECT_WITH_NEIGHBOR_INFO:
            for utm_id in so_utm_ids:
                for nei_utm_id in so_utm_ids:
                    if utm_id!=nei_utm_id:
                        if nei_utm_id not in \
                            self.mf_utm_id_to_mf_nei_utm_id_and_common_h_count\
                                            [utm_id]:
                            self.mf_utm_id_to_mf_nei_utm_id_and_common_h_count\
                                            [utm_id][nei_utm_id]=0.0
                        self.mf_utm_id_to_mf_nei_utm_id_and_common_h_count\
                                            [utm_id][nei_utm_id]+=1.0
    def map_final_hashtag_object_to_utm_object(self):
        for utm_id, mf_hashtag_to_count in \
                self.mf_utm_id_to_mf_hashtag_to_count.iteritems():
            utm_object = {
                           'mf_hashtag_to_count': mf_hashtag_to_count,
                           'total_hashtag_count': 
                                self.mf_utm_id_to_total_hashtag_count[utm_id]
                        }
            if UTM_OBJECT_WITH_NEIGHBOR_INFO:
                utm_object['mf_nei_utm_id_to_common_h_count'] =\
                    self.mf_utm_id_to_mf_nei_utm_id_and_common_h_count[utm_id]
            yield utm_id, utm_object
    def _get_valid_combined_utm_object(self, utm_id, utm_objects):
        combined_utm_object = {'utm_id': utm_id,
                               'mf_hashtag_to_count': defaultdict(float),
                               'total_hashtag_count' : 0.0,
                               'mf_nei_utm_id_to_common_h_count' : \
                                                            defaultdict(float)
                               }
        for utm_object in utm_objects:
            if utm_object['mf_hashtag_to_count']:
                mf_hashtag_to_count = utm_object['mf_hashtag_to_count']
                for hashtag, count in mf_hashtag_to_count.iteritems():
                    combined_utm_object['mf_hashtag_to_count'][hashtag]+=count
            if utm_object['total_hashtag_count']:
                combined_utm_object['total_hashtag_count']+=\
                                        utm_object['total_hashtag_count']
            if UTM_OBJECT_WITH_NEIGHBOR_INFO:
                for nei_utm_id, common_h_count in \
                        utm_object['mf_nei_utm_id_to_common_h_count'].\
                                                                    iteritems():
                    combined_utm_object\
                        ['mf_nei_utm_id_to_common_h_count'][nei_utm_id]\
                                                                +=common_h_count
        if UTM_OBJECT_WITH_NEIGHBOR_INFO:
            for nei_utm_id in combined_utm_object['mf_nei_utm_id_to_common_h_count'].keys()[:]:
                if combined_utm_object['mf_nei_utm_id_to_common_h_count'][nei_utm_id] < UTM_OBJECT_WITH_MIN_COMMON_HASHTAGS:
                    del combined_utm_object['mf_nei_utm_id_to_common_h_count'][nei_utm_id] 
        else: del combined_utm_object['mf_nei_utm_id_to_common_h_count']
        if combined_utm_object['total_hashtag_count'] >= \
                MIN_HASHTAG_OCCURRENCES_PER_UTM_ID:
            return combined_utm_object
    def red_tuo_utm_id_and_utm_objects_to_combined_utm_object(self,
                                                             utm_id,
                                                             utm_objects):
        combined_utm_object = self._get_valid_combined_utm_object(utm_id,
                                                                  utm_objects)
        if combined_utm_object: yield utm_id, combined_utm_object
    def jobs_to_get_utm_object(self):
        return self.hashtags_extractor.jobs_to_extract_hashtags() +\
                [self.mr(
                    mapper=self.map_hashtag_object_to_utm_object,
                    mapper_final=self.map_final_hashtag_object_to_utm_object,
                    reducer=
                    self.red_tuo_utm_id_and_utm_objects_to_combined_utm_object)
                ]
    def steps(self):
        return self.jobs_to_get_utm_object()
        
class HastagsWithUTMIdObject(ModifiedMRJob):
    '''
    hashtag_with_utm_object = {'hashtag' : hashtag,
                              'mf_utm_id_to_hashtag_occurrences': {
                                'total_num_of_occurrences' : 0,
                                  },
                            }
    '''
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(HastagsWithUTMIdObject, self).__init__(*args, **kwargs)
        self.hashtags_by_utm_id = HashtagsByUTMId()
        self.mf_hashtag_to_mf_utm_id_to_hashtag_occurrences = defaultdict(dict)
        
    def map_utm_object_to_hashtag_with_utm_object(self,
                                                      utm_id,
                                                      utm_object):
        if False: yield # I'm a generator!
        for hashtag, count in utm_object['mf_hashtag_to_count'].iteritems():
            if utm_id not in \
                    self.mf_hashtag_to_mf_utm_id_to_hashtag_occurrences\
                                                                    [hashtag]:
                self.mf_hashtag_to_mf_utm_id_to_hashtag_occurrences\
                        [hashtag][utm_id] = 0.0
            self.mf_hashtag_to_mf_utm_id_to_hashtag_occurrences\
                        [hashtag][utm_id]+=count
    def map_final_utm_object_to_hashtag_with_utm_object(self):
        for hashtag, mf_utm_id_to_hashtag_occurrences in \
                self.mf_hashtag_to_mf_utm_id_to_hashtag_occurrences.iteritems():
            mf_utm_id_to_hashtag_occurrences['total_num_of_occurrences'] = \
                sum(mf_utm_id_to_hashtag_occurrences.values())
            hashtag_with_utm_object = {'hashtag' : hashtag,
                                       'mf_utm_id_to_hashtag_occurrences': \
                                            mf_utm_id_to_hashtag_occurrences
                                    }
            yield hashtag, hashtag_with_utm_object
    def red_tuo_h_and_hashtag_with_utm_object_to_h_and_com_h_with_utm_object(
                                                 self,
                                                 hashtag,
                                                 hashtag_with_utm_objects):
        mf_utm_id_to_hashtag_occurrences = defaultdict(float)
        for hashtag_with_utm_object in hashtag_with_utm_objects:
            for utm_id, hashtag_occurrences in \
                    hashtag_with_utm_object\
                        ['mf_utm_id_to_hashtag_occurrences'].iteritems():
                mf_utm_id_to_hashtag_occurrences[utm_id]+=hashtag_occurrences
        combined_hashtag_with_utm_object = {'hashtag' : hashtag,
                                   'mf_utm_id_to_hashtag_occurrences': \
                                            mf_utm_id_to_hashtag_occurrences
                                }
        yield hashtag, combined_hashtag_with_utm_object
    def jobs_to_get_hashtags_with_utm_id_object(self):
        return self.hashtags_by_utm_id.jobs_to_get_utm_object() + \
                [self.mr(
                    mapper=self.map_utm_object_to_hashtag_with_utm_object,
                    mapper_final=
                        self.map_final_utm_object_to_hashtag_with_utm_object,
                    reducer=
                    self.red_tuo_h_and_hashtag_with_utm_object_to_h_and_com_h_with_utm_object)
                ]
    def steps(self):
        return self.jobs_to_get_hashtags_with_utm_id_object()
        
if __name__ == '__main__':
    pass
#    TweetStats.run()
#    HashtagsExtractor.run()
#    HashtagsDistributionInUTM.run()
    HashtagsByUTMId.run()
#    HastagsWithUTMIdObject.run()
