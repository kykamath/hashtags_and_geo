'''
Created on Sep 28, 2012

@author: krishnakamath
'''
from collections import defaultdict
from datetime import datetime
from library.classes import GeneralMethods
from library.geo import UTMConverter
from library.mrjobwrapper import ModifiedMRJob
#from library.r_helper import R_Helper
from library.stats import filter_outliers
from library.twitter import getDateTimeObjectFromTweetTimestamp
from itertools import chain, groupby
from operator import itemgetter
import cjson
import numpy as np
import time

ACCURACY = 10**4 # UTM boxes in sq.m

# Minimum number of hashtag occurrences
# Used by HashtagsExtractor
MIN_HASHTAG_OCCURRENCES = 250

## Minimum number of hashtag occurrences at a particular utm id.
## Used by HashtagsByUTMId
#MIN_HASHTAG_OCCURRENCES_PER_UTM_ID = 1250

# Start time for data analysis
START_TIME, END_TIME = datetime(2011, 3, 1), datetime(2012, 7, 31)

# Parameters for the MR Job that will be logged.
HASHTAG_STARTING_WINDOW = time.mktime(START_TIME.timetuple())
HASHTAG_ENDING_WINDOW = time.mktime(END_TIME.timetuple())

# Parameters for propagation analysis
MIN_HASHTAG_OCCURRENCES_FOR_PROPAGATION_ANALYSIS = 250
GAP_PERCT_FOR_PROPAGATION_ANALYSIS = 0.02
MIN_OCCURRENCES_PER_UTM_ID = 25
TIME_UNIT_IN_SECONDS = 60*10
MAJORITY_THRESHOLD_FOR_PROPAGATION_ANALYSIS = 0.15
        
PARAMS_DICT = dict(
                   PARAMS_DICT = True,
                   ACCURACY = ACCURACY,
                   MIN_HASHTAG_OCCURRENCES = MIN_HASHTAG_OCCURRENCES,
                   HASHTAG_STARTING_WINDOW = HASHTAG_STARTING_WINDOW,
                   HASHTAG_ENDING_WINDOW = HASHTAG_ENDING_WINDOW,
                   MIN_HASHTAG_OCCURRENCES_FOR_PROPAGATION_ANALYSIS = MIN_HASHTAG_OCCURRENCES_FOR_PROPAGATION_ANALYSIS,
                   MIN_OCCURRENCES_PER_UTM_ID = MIN_OCCURRENCES_PER_UTM_ID,
                   TIME_UNIT_IN_SECONDS = TIME_UNIT_IN_SECONDS,
                   )

def iterateHashtagObjectInstances(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    for h in data['h']: yield h.lower(), [l, t]
def get_items_at_gap(input_list, gap_perct):
    list_len = len(input_list)
    return map(
                  lambda index: input_list[index],
                  map(lambda index: int(index)-1, np.arange(gap_perct,1+gap_perct,gap_perct)*list_len)
              )

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
    
class PropagationMatrix(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(PropagationMatrix, self).__init__(*args, **kwargs)
        self.mf_perct_pair_to_time_differences = defaultdict(list)
    def mapper(self, key, line):
        if False: yield # I'm a generator!
        hashtag_object = cjson.decode(line)
        if 'num_of_occurrences' in hashtag_object and\
                hashtag_object['num_of_occurrences'] >= MIN_HASHTAG_OCCURRENCES_FOR_PROPAGATION_ANALYSIS:
            ltuo_occ_time_and_occ_utm_id = hashtag_object['ltuo_occ_time_and_occ_utm_id']
            ltuo_occ_time_and_occ_utm_id.sort(key=itemgetter(1))
            ltuo_occ_utm_id_and_occ_times =\
                [ (occ_utm_id,map(itemgetter(0), it_occ_time_and_occ_utm_id))
                 for occ_utm_id, it_occ_time_and_occ_utm_id in
                    groupby(ltuo_occ_time_and_occ_utm_id, key=itemgetter(1))
                ]
            ltuo_occ_utm_id_and_occ_times = filter(
                                                   lambda (_, occ_times): len(occ_times)>MIN_OCCURRENCES_PER_UTM_ID,
                                                   ltuo_occ_utm_id_and_occ_times
                                               )
            for occ_utm_id, occ_times in ltuo_occ_utm_id_and_occ_times:
                occ_times.sort()
                occ_times = filter_outliers(occ_times)
                lifespan = occ_times[-1] - occ_times[0]
                if lifespan > 0.0:
                    occ_times_at_gap_perct = get_items_at_gap(occ_times, GAP_PERCT_FOR_PROPAGATION_ANALYSIS)
                    ltuo_perct_and_occ_time = [
                                               (int((
                                                     GAP_PERCT_FOR_PROPAGATION_ANALYSIS*i+\
                                                        GAP_PERCT_FOR_PROPAGATION_ANALYSIS)*100),
                                                j)
                                                for i, j in enumerate(occ_times_at_gap_perct)
                                            ]
                    for perct1, occ_time1 in ltuo_perct_and_occ_time:
                        for perct2, occ_time2 in ltuo_perct_and_occ_time:
                            perct_pair = '%s_%s'%(perct1, perct2)
                            if perct2>perct1:
                                self.mf_perct_pair_to_time_differences[perct_pair].append(
                                                                              max(occ_time2-occ_time1, 0.0)
                                                                            )
                            else: self.mf_perct_pair_to_time_differences[perct_pair] = [0.0] 
    def mapper_final(self):
        for perct_pair, time_differences in self.mf_perct_pair_to_time_differences.iteritems():
            yield perct_pair, time_differences
    def reducer(self, perct_pair, it_time_differences):
        time_differences = list(chain(*it_time_differences))
        time_differences = filter_outliers(time_differences)
        yield perct_pair, {'perct_pair': perct_pair, 'time_differences': np.mean(time_differences)}

class HashtagsWithMajorityInfo(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(HashtagsWithMajorityInfo, self).__init__(*args, **kwargs)
    def mapper(self, key, value):
        hashtag_object = cjson.decode(value)
        if 'num_of_occurrences' in hashtag_object and\
                hashtag_object['num_of_occurrences'] >= MIN_HASHTAG_OCCURRENCES_FOR_PROPAGATION_ANALYSIS:
            ltuo_bucket_occ_time_and_occ_utm_id =\
                                        map(
                                               lambda (t, utm_id):
                                                    (GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS), utm_id),
                                               hashtag_object['ltuo_occ_time_and_occ_utm_id']
                                           )
            ltuo_bucket_occ_time_and_occ_utm_id.sort(key=itemgetter(1))
            ltuo_utm_id_and_bucket_occ_times =\
                [ (occ_utm_id,map(itemgetter(0), it_bucket_occ_time_and_occ_utm_id))
                 for occ_utm_id, it_bucket_occ_time_and_occ_utm_id in
                    groupby(ltuo_bucket_occ_time_and_occ_utm_id, key=itemgetter(1))
                ]
            ltuo_utm_id_and_bucket_occ_times =\
                                            filter(
                                                   lambda (_, occ_times): len(occ_times)>10,
                                                   ltuo_utm_id_and_bucket_occ_times
                                               )
            ltuo_utm_id_and_majority_threshold_bucket_time = []
            for utm_id, bucket_occ_times in ltuo_utm_id_and_bucket_occ_times:
                bucket_occ_times.sort()
                ltuo_utm_id_and_majority_threshold_bucket_time.append([
                               utm_id,
                               bucket_occ_times[int(MAJORITY_THRESHOLD_FOR_PROPAGATION_ANALYSIS*len(bucket_occ_times))]
                           ])
            ltuo_majority_threshold_bucket_time_and_utm_ids =\
                [ (majority_threshold_bucket_time, map(itemgetter(0), it_utm_id_and_majority_threshold_bucket_time))
                 for majority_threshold_bucket_time, it_utm_id_and_majority_threshold_bucket_time in
                    groupby(ltuo_utm_id_and_majority_threshold_bucket_time, itemgetter(1))
                ]
            ltuo_majority_threshold_bucket_time_and_utm_ids.sort(key=itemgetter(0))
            yield hashtag_object['hashtag'], {
                                              'hashtag': hashtag_object['hashtag'],
                                              'ltuo_majority_threshold_bucket_time_and_utm_ids':
                                                                        ltuo_majority_threshold_bucket_time_and_utm_ids
                                              }
            
if __name__ == '__main__':
    pass
#    HashtagsExtractor.run()
#    PropagationMatrix.run()
    HashtagsWithMajorityInfo.run()