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
from library.stats import MonteCarloSimulation
from library.stats import filter_outliers
from library.twitter import getDateTimeObjectFromTweetTimestamp
from itertools import chain, combinations, groupby
from operator import itemgetter
import cjson
import numpy as np
import random
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
                   GAP_PERCT_FOR_PROPAGATION_ANALYSIS = GAP_PERCT_FOR_PROPAGATION_ANALYSIS
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
                                              'num_of_occurrences': hashtag_object['num_of_occurrences'],
                                              'ltuo_majority_threshold_bucket_time_and_utm_ids':
                                                                        ltuo_majority_threshold_bucket_time_and_utm_ids
                                              }

class HashtagsWithMajorityInfoAtVaryingGaps(ModifiedMRJob):
    '''
    hashtag_with_majority_info_object = {
                                'hashtag': hashtag,
                                'ltuo_majority_threshold_bucket_time_and_utm_ids':
                                                                        ltuo_majority_threshold_bucket_time_and_utm_ids
                            }
    '''
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(HashtagsWithMajorityInfoAtVaryingGaps, self).__init__(*args, **kwargs)
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
            for gap in [0.15, 0.25, 0.50, 0.75]:
                ltuo_utm_id_and_majority_threshold_bucket_time = []
                for utm_id, bucket_occ_times in ltuo_utm_id_and_bucket_occ_times:
                    bucket_occ_times.sort()
                    ltuo_utm_id_and_majority_threshold_bucket_time.append([
                                   utm_id,
                                   bucket_occ_times[int(gap*len(bucket_occ_times))]
                               ])
                ltuo_majority_threshold_bucket_time_and_utm_ids =\
                    [ (majority_threshold_bucket_time, map(itemgetter(0), it_utm_id_and_majority_threshold_bucket_time))
                     for majority_threshold_bucket_time, it_utm_id_and_majority_threshold_bucket_time in
                        groupby(ltuo_utm_id_and_majority_threshold_bucket_time, itemgetter(1))
                    ]
                ltuo_majority_threshold_bucket_time_and_utm_ids.sort(key=itemgetter(0))
                yield '%0.2f'%gap, {
                                      'hashtag': hashtag_object['hashtag'],
                                      'ltuo_majority_threshold_bucket_time_and_utm_ids':
                                                            ltuo_majority_threshold_bucket_time_and_utm_ids
                                  }
    def reducer(self, gap_id, ito_hashtag_with_majority_info_object):
        hashtag_with_majority_info_objects = []
#        hashtag_with_majority_info_objects = list(chain(*ito_hashtag_with_majority_info_object))
        for hashtag_with_majority_info_object in ito_hashtag_with_majority_info_object:
            hashtag_with_majority_info_objects.append(hashtag_with_majority_info_object)
        yield gap_id, {
                           'gap_id': gap_id,
                           'hashtag_with_majority_info_objects': hashtag_with_majority_info_objects
                       }
        

def group_items_by(list_object, key):
    list_object.sort(key=key)
    return [(k,list(ito_items)) for k, ito_items in groupby(list_object, key=key)]
            
class ImpactOfUsingLocationsToPredict(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
#    STATUS_TOGETHER = 0
    STATUS_BEFORE = -1
    STATUS_AFTER = 1
#    MIN_COMMON_HASHTAGS = [1,2]
    MIN_COMMON_HASHTAGS = range(5,101,5)
    def __init__(self, *args, **kwargs):
        super(ImpactOfUsingLocationsToPredict, self).__init__(*args, **kwargs)
        self.mf_location_pair_to_propagation_statuses = defaultdict(list)
    def mapper(self, key, value):
        if False: yield # I'm a generator!
        hashtag_object = cjson.decode(value)
        if 'num_of_occurrences' in hashtag_object and\
                hashtag_object['num_of_occurrences'] >= MIN_HASHTAG_OCCURRENCES_FOR_PROPAGATION_ANALYSIS:
            ltuo_occ_time_and_occ_utm_id = hashtag_object['ltuo_occ_time_and_occ_utm_id']
            ltuo_utm_id_and_ltuo_occ_time_and_occ_utm_id =\
                                                        group_items_by(ltuo_occ_time_and_occ_utm_id, key=itemgetter(1))
            ltuo_utm_id_and_occ_times = map(
                                            lambda (u, l_o_u): (u, map(itemgetter(0), l_o_u)),
                                            ltuo_utm_id_and_ltuo_occ_time_and_occ_utm_id
                                        )
            ltuo_utm_id_and_occ_times = filter(
                                               lambda (u, l_o): len(l_o) >= MIN_OCCURRENCES_PER_UTM_ID,
                                               ltuo_utm_id_and_occ_times
                                               )
            ltuo_utm_id_and_occ_times = map(lambda (u,o): (u,sorted(o)), ltuo_utm_id_and_occ_times)
            ltuo_utm_id_and_majority_threshold_bucket_time =\
                                map(
                                     lambda (u, l_o):(
                                            u,
                                            GeneralMethods.approximateEpoch(
                                                        l_o[int(MAJORITY_THRESHOLD_FOR_PROPAGATION_ANALYSIS*len(l_o))],
                                                        TIME_UNIT_IN_SECONDS
                                            )
                                     ),
                                     ltuo_utm_id_and_occ_times
                                 )
            for u_and_t1, u_and_t2 in combinations(ltuo_utm_id_and_majority_threshold_bucket_time, 2):
                smaller_u_and_t, bigger_u_and_t = sorted([u_and_t1, u_and_t2], key=itemgetter(0))
                location_pair = '%s::%s'%(smaller_u_and_t[0], bigger_u_and_t[0])
#                propagation_status = ImpactOfUsingLocationsToPredict.STATUS_TOGETHER
                propagation_status = None
                if smaller_u_and_t[1] < bigger_u_and_t[1]:
                    propagation_status = ImpactOfUsingLocationsToPredict.STATUS_BEFORE
                elif smaller_u_and_t[1] > bigger_u_and_t[1]:
                    propagation_status = ImpactOfUsingLocationsToPredict.STATUS_AFTER
                if propagation_status:
                    self.mf_location_pair_to_propagation_statuses[location_pair].append(propagation_status)
    def mapper_final(self):
        for location_pair, propagation_statuses in self.mf_location_pair_to_propagation_statuses.iteritems():
            yield location_pair, propagation_statuses
    def reducer(self, location_pair, it_propagation_statuses):
        propagation_statuses = list(chain(*it_propagation_statuses))
        for min_common_hashtag in ImpactOfUsingLocationsToPredict.MIN_COMMON_HASHTAGS:
            if len(propagation_statuses) > min_common_hashtag:
                yield min_common_hashtag, np.mean(propagation_statuses)
            else: break
    def reducer_with_monte_carlo_simulation(self, location_pair, it_propagation_statuses):
        propagation_statuses = list(chain(*it_propagation_statuses))
        for min_common_hashtag in ImpactOfUsingLocationsToPredict.MIN_COMMON_HASHTAGS:
            if len(propagation_statuses) > min_common_hashtag:
                mean_probability = MonteCarloSimulation.mean_probability(
                                                 MonteCarloSimulation.probability_of_data_extracted_from_same_sample,
                                                 propagation_statuses,
                                                 [random.sample([
                                                                 ImpactOfUsingLocationsToPredict.STATUS_BEFORE,
                                                                 ImpactOfUsingLocationsToPredict.STATUS_AFTER
                                                                 ],
                                                                1)[0] 
                                                  for i in range(len(propagation_statuses))]
                                             )
                yield min_common_hashtag, {
                                           'location_pair': location_pair,
                                           'mean_probability': mean_probability,
                                           'len_propagation_statuses': len(propagation_statuses),
                                           'propagation_statuses': np.mean(propagation_statuses)
                                           }
            else: break
    def reducer2(self, min_common_hashtag, it_mean_propagation_statuses):
        yield min_common_hashtag, {
                                   'min_common_hashtag':min_common_hashtag, 
                                   'mean_propagation_statuses': list(it_mean_propagation_statuses)
                                }
    def jobs_for_mean_impact_values(self):
        return [
                self.mr(self.mapper, self.reducer, self.mapper_final),
                self.mr(reducer=self.reducer2)
                ]
    def jobs_for_analyzing_impact_using_mc_simulation(self):
        return [
                self.mr(self.mapper, self.reducer_with_monte_carlo_simulation, self.mapper_final),
                self.mr(reducer=self.reducer2)
                ]
    def steps(self):
        return self.jobs_for_analyzing_impact_using_mc_simulation()
    
class GapOccurrenceTimeDuringHashtagLifetime(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(GapOccurrenceTimeDuringHashtagLifetime, self).__init__(*args, **kwargs)
        self.mf_gap_to_norm_num_of_occurrences = defaultdict(float)
    def mapper(self, key, value):
        if False: yield # I'm a generator!
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
            for _, bucket_occ_times in ltuo_utm_id_and_bucket_occ_times:
                gap_perct = 0.05
                gaps = np.arange(gap_perct,1+gap_perct,gap_perct)
                bucket_occ_times = filter_outliers(bucket_occ_times)
                bucket_occ_times_at_gaps = get_items_at_gap(bucket_occ_times, gap_perct)
                start_time = float(bucket_occ_times_at_gaps[0])
                life_time = bucket_occ_times_at_gaps[-1] - start_time
                if life_time>0:
                    norm_num_of_occurrences =\
                                            map(lambda t: int(((t-start_time)/life_time)*100), bucket_occ_times_at_gaps)
                    for gap, norm_num_of_occurrence in zip(gaps, norm_num_of_occurrences):
                        self.mf_gap_to_norm_num_of_occurrences['%0.2f'%gap]+=norm_num_of_occurrence
    def mapper_final(self): yield '', self.mf_gap_to_norm_num_of_occurrences.items()
    def reducer(self, empty_key, it_ltuo_gap_and_norm_num_of_occurrences):
        mf_gap_to_norm_num_of_occurrences = defaultdict(float)
        for ltuo_gap_and_norm_num_of_occurrences in it_ltuo_gap_and_norm_num_of_occurrences:
            for gap, norm_num_of_occurrences in ltuo_gap_and_norm_num_of_occurrences:
                mf_gap_to_norm_num_of_occurrences[gap]+=norm_num_of_occurrences
        total_num_of_occurrences = sum(mf_gap_to_norm_num_of_occurrences.values())
        ltuo_gap_and_perct_of_occurrences =\
                map(lambda (g, n): (float(g), n/total_num_of_occurrences), mf_gap_to_norm_num_of_occurrences.items())
        ltuo_gap_and_perct_of_occurrences.sort(key=itemgetter(0))
        yield '', ltuo_gap_and_perct_of_occurrences

class LocationClusters(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(LocationClusters, self).__init__(*args, **kwargs)
        self.mf_utm_id_to_ltuo_hashtag_and_neighbor_utm_ids = defaultdict(list)
    def mapper(self, key, value):
        if False: yield # I'm a generator!
        hashtag_object = cjson.decode(value)
        if 'num_of_occurrences' in hashtag_object and\
                hashtag_object['num_of_occurrences'] >= MIN_HASHTAG_OCCURRENCES_FOR_PROPAGATION_ANALYSIS:
            ltuo_occ_time_and_occ_utm_id = hashtag_object['ltuo_occ_time_and_occ_utm_id']
            total_hashtag_occurrences = len(ltuo_occ_time_and_occ_utm_id)
            ltuo_occ_utm_id_and_occ_counts = map(
                                                     lambda (u, occ): (u, len(occ)), 
                                                     group_items_by(ltuo_occ_time_and_occ_utm_id, key=itemgetter(1))
                                                 )
            ltuo_occ_utm_id_and_occ_counts = filter(
                                                        lambda (u, c): c>=(0.03*total_hashtag_occurrences),
                                                        ltuo_occ_utm_id_and_occ_counts
                                                    )
            if ltuo_occ_utm_id_and_occ_counts:
                utm_ids = zip(*ltuo_occ_utm_id_and_occ_counts)[0]
                for utm_id in utm_ids:
                    self.mf_utm_id_to_ltuo_hashtag_and_neighbor_utm_ids[utm_id]\
                                                                        .append([hashtag_object['hashtag'], utm_ids])
    def mapper_final(self):
        for utm_id, ltuo_hashtag_and_neighbor_utm_ids in\
                self.mf_utm_id_to_ltuo_hashtag_and_neighbor_utm_ids.iteritems():
            yield utm_id, ltuo_hashtag_and_neighbor_utm_ids
    def reducer(self, utm_id, it_ltuo_hashtag_and_neighbor_utm_ids):
        ltuo_hashtag_and_neighbor_utm_ids = list(chain(*it_ltuo_hashtag_and_neighbor_utm_ids))
        hashtags, neighbor_utm_ids = zip(*ltuo_hashtag_and_neighbor_utm_ids)
        neighbor_utm_ids = list(set(chain(*neighbor_utm_ids)))
        utm_id_and_hashtags = [utm_id, hashtags]
        if len(hashtags) >= 25:
            for neighbor_utm_id in neighbor_utm_ids: yield neighbor_utm_id, utm_id_and_hashtags
            yield utm_id, utm_id_and_hashtags
    def reducer2(self, utm_id, it_utm_id_and_hashtags):
        ltuo_neighbor_utm_id_and_neighbor_hashtags = []
        hashtags = None
        for neighbor_utm_id, neighbor_hashtags in it_utm_id_and_hashtags:
            if neighbor_utm_id == utm_id: hashtags = set(neighbor_hashtags)
            elif utm_id<neighbor_utm_id:
                ltuo_neighbor_utm_id_and_neighbor_hashtags.append([neighbor_utm_id, set(neighbor_hashtags)])
        if hashtags:
            for neighbor_utm_id, neighbor_hashtags in ltuo_neighbor_utm_id_and_neighbor_hashtags:
                num_common_hashtags = len(hashtags.intersection(neighbor_hashtags))+0.0
                total_hashtags = len(hashtags.union(neighbor_hashtags))
                if num_common_hashtags/total_hashtags >= 0.25:
                    observed_hashtag_pattern = [1 for i in range(num_common_hashtags)] +\
                                                                    [0 for i in range(total_hashtags - num_common_hashtags)]
                    mean_probability = np.mean([
                                                MonteCarloSimulation.mean_probability(
                                                         MonteCarloSimulation.probability_of_data_extracted_from_same_sample,
                                                         observed_hashtag_pattern,
                                                         [random.sample([0,1], 1)[0] for i in range(total_hashtags)]
                                                     )
                                               for i in range(3)])
#                    print utm_id, neighbor_utm_id
#                    print observed_hashtag_pattern, mean_probability
#                    print [random.sample([0,1], 1)[0] for i in range(total_hashtags)]
                    if mean_probability <= 0.05: yield '', {
                                                            'utm_id': utm_id,
                                                            'neighbor_utm_id': neighbor_utm_id,
                                                            'mean_probability':mean_probability,
                                                            'num_common_hashtags': num_common_hashtags
                                                        }
    def steps(self):
        return [
                    self.mr(mapper=self.mapper, reducer=self.reducer, mapper_final=self.mapper_final),
                    self.mr(reducer=self.reducer2)
                ]
if __name__ == '__main__':
    pass
#    HashtagsExtractor.run()
#    PropagationMatrix.run()
#    HashtagsWithMajorityInfo.run()
#    HashtagsWithMajorityInfoAtVaryingGaps.run()
#    ImpactOfUsingLocationsToPredict.run()
#    GapOccurrenceTimeDuringHashtagLifetime.run()
    LocationClusters.run()
    
