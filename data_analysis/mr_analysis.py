'''
Created on May 7, 2012

@author: krishnakamath
'''
from itertools import combinations
import cjson, time
from library.mrjobwrapper import ModifiedMRJob
from library.twitter import getDateTimeObjectFromTweetTimestamp
from library.geo import getLatticeLid, getLocationFromLid, getRadiusOfGyration
from library.classes import GeneralMethods
from collections import defaultdict
from datetime import datetime
from library.stats import entropy, focus
from operator import itemgetter
import numpy as np

LOCATION_ACCURACY = 1.45 # 100 miles
#LOCATION_ACCURACY = 0.001 # 100 miles

TIME_UNIT_IN_SECONDS = 60*10 # 10 minutes Used for iid only
#TIME_UNIT_IN_SECONDS = 60*60 # 60 minutes Used for norm iid to overcome sparcity

MIN_HASHTAG_OCCURENCES = 50
#MAX_HASHTAG_OCCURENCES = 100000
START_TIME, END_TIME = datetime(2011, 3, 1), datetime(2012, 3, 31)
#START_TIME, END_TIME = datetime(2011, 3, 1), datetime(2011, 4, 28)


#Distribution Paramter
DISTRIBUTION_ACCURACY = 100

# Top K rank analysis
#K_TOP_RANK = 100
K_TOP_RANK = 11935

# Temporal analysis
VALID_IID_RANGE = range(-30,31)
MIN_NUMBER_OF_SHARED_HASHTAGS = 10

# Parameters for the MR Job that will be logged.
HASHTAG_STARTING_WINDOW, HASHTAG_ENDING_WINDOW = time.mktime(START_TIME.timetuple()), time.mktime(END_TIME.timetuple())
PARAMS_DICT = dict(PARAMS_DICT = True,
                   LOCATION_ACCURACY=LOCATION_ACCURACY,
                   MIN_HASHTAG_OCCURENCES=MIN_HASHTAG_OCCURENCES,
                   TIME_UNIT_IN_SECONDS = TIME_UNIT_IN_SECONDS,
                   HASHTAG_STARTING_WINDOW = HASHTAG_STARTING_WINDOW, HASHTAG_ENDING_WINDOW = HASHTAG_ENDING_WINDOW,
                   K_TOP_RANK = K_TOP_RANK,
                   MIN_NUMBER_OF_SHARED_HASHTAGS = MIN_NUMBER_OF_SHARED_HASHTAGS,
                   )

def iterate_hashtag_occurrences(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    lid = getLatticeLid(l, LOCATION_ACCURACY)
    for h in data['h']: yield h.lower(), [lid, GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS)]

def iterate_hashtag_occurrences_with_high_accuracy_lid(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    lid = getLatticeLid(l, accuracy=0.0001)
    for h in data['h']: yield h.lower(), [lid, GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS)]
    
def combine_hashtag_instances(hashtag, ito_ltuo_lid_and_occurrence_time):
    combined_ltuo_lid_and_occurrence_time = []
    for ltuo_lid_and_occurrence_time in ito_ltuo_lid_and_occurrence_time: 
        for tuo_lid_and_occurrence_time in ltuo_lid_and_occurrence_time: 
            combined_ltuo_lid_and_occurrence_time.append(tuo_lid_and_occurrence_time)
    if combined_ltuo_lid_and_occurrence_time:
        e, l = min(combined_ltuo_lid_and_occurrence_time, key=lambda t: t[1]), max(combined_ltuo_lid_and_occurrence_time, key=lambda t: t[1])
        if len(combined_ltuo_lid_and_occurrence_time)>=MIN_HASHTAG_OCCURENCES and \
                e[1]>=HASHTAG_STARTING_WINDOW and l[1]<=HASHTAG_ENDING_WINDOW:
            return {
                    'hashtag': hashtag, 
                    'ltuo_lid_and_s_interval': sorted(combined_ltuo_lid_and_occurrence_time, key=lambda t: t[1])
                    }

def get_mf_interval_to_mf_lid_to_occurrence_count(hashtag_object):
    mf_interval_to_mf_lid_to_occurrence_count = defaultdict(dict)
    for lid, interval in hashtag_object['ltuo_lid_and_s_interval']:
        interval = str(interval)
        if lid not in \
                mf_interval_to_mf_lid_to_occurrence_count[interval]:
            mf_interval_to_mf_lid_to_occurrence_count[interval][lid] = 0.
        mf_interval_to_mf_lid_to_occurrence_count[interval][lid]+=1
    return mf_interval_to_mf_lid_to_occurrence_count

def get_mf_lid_to_occurrence_count(hashtag_object):
    return_mf_lid_to_occurrence_count = defaultdict(float)
    mf_interval_to_mf_lid_to_occurrence_count = \
        get_mf_interval_to_mf_lid_to_occurrence_count(hashtag_object)
    for interval, mf_lid_to_occurrence_count in \
            mf_interval_to_mf_lid_to_occurrence_count.iteritems():
        for lid, occurrence_count in \
                mf_lid_to_occurrence_count.iteritems():
            return_mf_lid_to_occurrence_count[lid]+=occurrence_count 
    return return_mf_lid_to_occurrence_count

def get_ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count(hashtag_object):
    return_ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count = []
    ltuo_interval_and_mf_lid_to_occurrence_count = \
        get_mf_interval_to_mf_lid_to_occurrence_count(hashtag_object).items()
    ltuo_s_interval_and_mf_lid_to_occurrence_count = sorted(ltuo_interval_and_mf_lid_to_occurrence_count, key=itemgetter(0))
    first_interval = ltuo_s_interval_and_mf_lid_to_occurrence_count[0][0]
    first_interval=int(float(first_interval))
    for interval, mf_lid_to_occurrence_count in \
            ltuo_s_interval_and_mf_lid_to_occurrence_count:
        interval=int(float(interval))
        iid = (interval-first_interval)/TIME_UNIT_IN_SECONDS
        return_ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count.append(
                  [iid, [interval, mf_lid_to_occurrence_count.items()]]                            
                  )
    return return_ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count

def get_ltuo_iid_and_tuo_interval_and_occurrence_count(hashtag_object):
    return_ltuo_iid_to_tuo_interval_and_occurrence_count = []
    ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count = \
        get_ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count(hashtag_object)
    for iid, (interval, ltuo_lid_and_occurrence_count) in\
            ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count:
        return_ltuo_iid_to_tuo_interval_and_occurrence_count.append(
                    [iid, [interval, sum(zip(*ltuo_lid_and_occurrence_count)[1])]]
                )
    return return_ltuo_iid_to_tuo_interval_and_occurrence_count

def get_ltuo_iid_and_tuo_interval_and_lids(hashtag_object):
    return_ltuo_iid_and_tuo_interval_and_lids = []
    ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count = \
        get_ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count(hashtag_object)
    for iid, (interval, ltuo_lid_and_occurrence_count) in\
            ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count:
        lids = []
        for lid, occurrence_count in ltuo_lid_and_occurrence_count:
            for i in range(occurrence_count): lids.append(lid)
        return_ltuo_iid_and_tuo_interval_and_lids.append(
                    [iid, [interval, lids]]
                )
    return return_ltuo_iid_and_tuo_interval_and_lids

def get_ltuo_valid_iid_and_focus_lid(hashtag_object):
    # Get peak
    ltuo_iid_and_tuo_interval_and_lids = \
        get_ltuo_iid_and_tuo_interval_and_lids(hashtag_object)
    peak_tuo_iid_and_tuo_interval_and_lids = \
        max(ltuo_iid_and_tuo_interval_and_lids, key=lambda (_, (__, lids)): len(lids))
    peak_iid = peak_tuo_iid_and_tuo_interval_and_lids[0]
    # Get valid intervals with corresponding focus lids
    ltuo_valid_iid_and_focus_lid = []
    ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count = \
        get_ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count(hashtag_object)
    so_observed_focus_lids = set()
    for iid, (interval, ltuo_lid_and_occurrence_count) in \
            ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count:
        if (iid-peak_iid) in VALID_IID_RANGE: 
            focus_lid  = focus(dict(ltuo_lid_and_occurrence_count))[0]
            if focus_lid not in so_observed_focus_lids:
                ltuo_valid_iid_and_focus_lid.append([iid, focus_lid])
                so_observed_focus_lids.add(focus_lid)
    return ltuo_valid_iid_and_focus_lid
def get_so_observed_focus_lids(hashtag_object):
    ltuo_valid_iid_and_focus_lid = get_ltuo_valid_iid_and_focus_lid(hashtag_object)
    return set(zip(*ltuo_valid_iid_and_focus_lid)[1])
#    # Get peak
#    ltuo_iid_and_tuo_interval_and_lids = \
#        get_ltuo_iid_and_tuo_interval_and_lids(hashtag_object)
#    peak_tuo_iid_and_tuo_interval_and_lids = \
#        max(ltuo_iid_and_tuo_interval_and_lids, key=lambda (_, (__, lids)): len(lids))
#    peak_iid = peak_tuo_iid_and_tuo_interval_and_lids[0]
#    # Get valid intervals with corresponding focus lids
#    ltuo_valid_iid_and_focus_lid = []
#    ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count = \
#        get_ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count(hashtag_object)
#    so_observed_focus_lids = set()
#    for iid, (interval, ltuo_lid_and_occurrence_count) in \
#            ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count:
#        if (iid-peak_iid) in VALID_IID_RANGE: 
#            focus_lid  = focus(dict(ltuo_lid_and_occurrence_count))[0]
#            if focus_lid not in so_observed_focus_lids:
#                ltuo_valid_iid_and_focus_lid.append([iid, focus_lid])
#                so_observed_focus_lids.add(focus_lid)
#    return so_observed_focus_lids

#def get_ltuo_iid_and_tuo_interval_and_occurrences(hashtag_object):
#    return_ltuo_iid_and_tuo_interval_and_occurrences = []
#    mf_interval_to_mf_lid_to_occurrences = defaultdict(dict)
##    mf_interval_to_mf_lid_to_occurrence_count = defaultdict(dict)
#    for lid, interval in hashtag_object['ltuo_lid_and_s_interval']:
##        interval = str(interval)
#        if lid not in mf_interval_to_mf_lid_to_occurrences[interval]:
#            mf_interval_to_mf_lid_to_occurrences[interval][lid] = []
#        mf_interval_to_mf_lid_to_occurrences[interval][lid].append(lid)
#    return mf_interval_to_mf_lid_to_occurrence_count

class MRAnalysis(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(MRAnalysis, self).__init__(*args, **kwargs)
        self.mf_hashtag_to_ltuo_lid_and_occurrence_time = defaultdict(list)
        self.mf_hashtag_to_occurrence_count = defaultdict(float)
        # Stat variables
        self.number_of_tweets = 0.0
        self.number_of_geo_tweets = 0.0
        self.so_hashtags = set()
        self.so_lids = set()
        # Variables for tuo_lid_and_distribution_value
        self.mf_lid_to_occurrence_count = defaultdict(float)
        # High accuracy lid
        self.mf_high_accuracy_lid_to_count = defaultdict(float)
    
    
    ''' Start: Methods to load hashtag objects
    '''
    def map_checkin_line_to_tuo_hashtag_and_ltuo_lid_and_occurrence_time(self, key, line):
        if False: yield # I'm a generator!
        for h, tuo_lid_and_occurrence_time in iterate_hashtag_occurrences(line): 
            self.mf_hashtag_to_ltuo_lid_and_occurrence_time[h].append(tuo_lid_and_occurrence_time)
    def mapf_checkin_line_to_tuo_hashtag_and_ltuo_lid_and_occurrence_time(self):
        for h, ltuo_lid_and_occurrence_time in \
                self.mf_hashtag_to_ltuo_lid_and_occurrence_time.iteritems(): # e = earliest, l = latest
            yield h, ltuo_lid_and_occurrence_time
    def red_tuo_hashtag_and_ito_ltuo_lid_and_occurrence_time_to_tuo_hashtag_and_hashtag_object(self, hashtag, ito_ltuo_lid_and_occurrence_time):
        hashtagObject = combine_hashtag_instances(hashtag, ito_ltuo_lid_and_occurrence_time)
        if hashtagObject: yield hashtag, hashtagObject 
    ''' End: Methods to load hashtag objects
    '''
    
    ''' Start: Methods to get distribution in high accuracy lid
    '''
    def map_checkin_line_to_tuo_high_accuracy_lid_and_ltuo_lid_and_count(self, key, line):
        if False: yield # I'm a generator!
        for h, tuo_lid_and_occurrence_time in iterate_hashtag_occurrences_with_high_accuracy_lid(line): 
            self.mf_high_accuracy_lid_to_count[tuo_lid_and_occurrence_time[0]]+=1
    def mapf_checkin_line_to_tuo_high_accuracy_lid_and_ltuo_lid_and_count(self):
        for high_accuracy_lid, count in \
                self.mf_high_accuracy_lid_to_count.iteritems(): # e = earliest, l = latest
            yield high_accuracy_lid, count
    def red_tuo_lid_and_ito_count_and_occurrence_time_to_tuo_lid_and_count(self, high_accuracy_lid, ito_count):
        red_count = 0.0
        for count in ito_count: red_count+=count
        if red_count>500: yield high_accuracy_lid, [high_accuracy_lid, red_count]
    ''' End: Methods to get distribution in high accuracy lid
    '''
        
    ''' Start: Methods to load preprocessed hashtag objects
    '''    
    def map_hashtag_object_string_to_tuo_of_hashtag_and_hashtag_object(self, key, hashtag_object_line):
        hashtag_object = cjson.decode(hashtag_object_line)
        if 'hashtag' in hashtag_object: yield hashtag_object['hashtag'], hashtag_object
    def red_tuo_hashtag_and_ito_hashtag_object_to_tuo_hashtag_and_hashtag_object(self, hashtag, ito_hashtag_object):
        yield hashtag, list(ito_hashtag_object)[0]
    ''' End: Methods to load preprocessed hashtag objects
    '''
        
    ''' Start: Methods to determine hashtag distribution
    '''    
    def map_hashtag_object_to_tuo_no_of_hashtags_and_count(self, hashtag, hashtag_object):
        yield len(hashtag_object['ltuo_lid_and_s_interval']), 1.
    def red_tuo_no_of_hashtags_and_ito_count_to_no_of_hashtags_and_count(self, no_of_hashtags, ito_count):
        red_count = 1.0
        for count in ito_count: red_count+=count
        yield no_of_hashtags, [no_of_hashtags, red_count]
    ''' End: Methods to determine hashtag distribution
    '''
        
    ''' Start: Methods to determine hashtag spatial distribution
    '''    
    def map_hashtag_object_to_tuo_no_of_locations_and_count(self, hashtag, hashtag_object):
        lids = zip(*hashtag_object['ltuo_lid_and_s_interval'])[0]
        yield len(set(lids)), 1.
    def red_tuo_no_of_locations_and_ito_count_to_no_of_locations_and_count(self, no_of_locations, ito_count):
        red_count = 1.0
        for count in ito_count: red_count+=count
        yield no_of_locations, [no_of_locations, red_count]
    ''' End: Methods to determine hashtag spatial distribution
    '''
        
    
    ''' Start: Methods to get total tweets and total geo tweets
    '''
    def map_checkin_line_to_tuo_stat_and_stat_value(self, key, line):
        if False: yield # I'm a generator!
        self.number_of_tweets+=1
        flag = False
        for h, (lid, occurrence_time) in iterate_hashtag_occurrences(line): 
            self.so_hashtags.add(h), self.so_lids.add(lid)
            flag=True
        if flag: self.number_of_geo_tweets+=1
    def mapf_checkin_line_to_tuo_stat_and_stat_value(self):
        yield 'number_of_geo_tweets', self.number_of_tweets
        yield 'number_of_geo_tweets_with_hashtag', self.number_of_geo_tweets
        yield 'hashtags', [hashtag for hashtag in self.so_hashtags]
        yield 'lids', [lid for lid in self.so_lids]
    def red_tuo_stat_and_ito_stat_value_to_tuo_stat_and_stat_value(self, stat, ito_stat_value):
        if stat!='hashtags' and stat!='lids':
            reduced_stat_value = 0.0
            for stat_value in ito_stat_value: reduced_stat_value+=stat_value
            yield stat, [stat, reduced_stat_value]
        else:
            reduced_so_stat_vals = set()
            for stat_vals in ito_stat_value: 
                for stat_val in stat_vals: reduced_so_stat_vals.add(stat_val)
            yield stat, [stat, len(reduced_so_stat_vals)]
    ''' End: Methods to get total tweets and total geo tweets
    '''
    
    
    ''' Start: Methods to get hashtag occurrence distribution
    '''
    def map_checkin_line_to_tuo_hashtag_and_occurrence_count(self, key, line):
        if False: yield # I'm a generator!
        for h, tuo_lid_and_occurrence_time in iterate_hashtag_occurrences(line): 
            self.mf_hashtag_to_occurrence_count[h]+=1
    def mapf_checkin_line_to_tuo_hashtag_and_occurrence_count(self):
        for h, occurrence_count in \
                self.mf_hashtag_to_occurrence_count.iteritems():
            yield h, occurrence_count
    def red_tuo_hashtag_and_ito_occurrence_count_to_tuo_normalized_occurrence_count_and_one(self, hashtag, ito_occurrence_count):
        occurrences_count = 0.0
        for occurrence_count in ito_occurrence_count: occurrences_count+=occurrence_count
        yield int(occurrences_count/DISTRIBUTION_ACCURACY)*DISTRIBUTION_ACCURACY+DISTRIBUTION_ACCURACY, 1.0
    def red_tuo_normalized_occurrence_count_and_ito_one_to_tuo_normalized_occurrence_count_and_distribution_value(self, normalized_occurrence_count, ito_one):
        distribution_value = 1.0
        for one in ito_one: distribution_value+=one
        yield normalized_occurrence_count, [normalized_occurrence_count, distribution_value]
    ''' End: Methods to get hashtag occurrence distribution
    '''
    
    
    ''' Start: Methods to get distribution of occurrences in lids
    '''
    def map_checkin_line_to_tuo_lid_and_occurrence_count(self, key, line):
        if False: yield # I'm a generator!
        for h, (lid, occurrence_time) in iterate_hashtag_occurrences(line): 
            self.mf_lid_to_occurrence_count[lid]+=1
    def mapf_checkin_line_to_tuo_lid_and_occurrence_count(self):
        for lid, occurrence_count in\
                self.mf_lid_to_occurrence_count.iteritems():
            yield lid, occurrence_count
    def red_tuo_lid_and_ito_occurrence_count_to_tuo_lid_and_occurrence_count(self, lid, ito_occurrence_count):
        red_occurrence_count = 0.0
        for occurrence_count in ito_occurrence_count:
            red_occurrence_count+=occurrence_count
        yield lid, [lid, red_occurrence_count]
    ''' End: Methods to get distribution of occurrences in lids
    '''
    
    ''' Start: Methods to get distribution from top-k hashtags
    '''
    def map_hashtag_object_to_tuo_cum_prct_of_occurrences_and_prct_of_occurrences(self, hashtag, hashtag_object):
        mf_lid_to_occurrence_count = get_mf_lid_to_occurrence_count(hashtag_object)
        ltuo_lid_and_r_occurrence_count = sorted(mf_lid_to_occurrence_count.items(), key=itemgetter(1), reverse=True)
        total_occurrence_count = float(sum(zip(*ltuo_lid_and_r_occurrence_count)[1]))
        current_occurrence_count = 0
#        for rank, (_, occurrence_count) in enumerate(ltuo_lid_and_r_occurrence_count[:K_TOP_RANK]):
#            current_occurrence_count+=occurrence_count
#            yield rank+1, current_occurrence_count/total_occurrence_count
        rank = 0
        for _, occurrence_count in ltuo_lid_and_r_occurrence_count:
            current_occurrence_count+=occurrence_count
            rank+=1
            yield rank, [current_occurrence_count/total_occurrence_count, occurrence_count/total_occurrence_count]
        # Setting ranks for extra locations to 1.
        while rank < K_TOP_RANK:
            rank+=1
            yield rank, [1.0, 0.0]
            
    def red_tuo_rank_and_ito_percentage_of_occurrences_to_tuo_rank_and_avg_cum_prct_of_occurrences_and_avg_prct_of_occurrences(self, rank, ito_cum_prct_of_occurrences_and_prct_of_occurrences):
        red_cum_prct_of_occurrences = []
        red_prct_of_occurrences = []
        for (cum_prct_of_occurrences, prct_of_occurrences) in ito_cum_prct_of_occurrences_and_prct_of_occurrences: 
            red_cum_prct_of_occurrences.append(cum_prct_of_occurrences)
            red_prct_of_occurrences.append(prct_of_occurrences)
        yield rank, [rank, np.mean(red_cum_prct_of_occurrences), np.mean(red_prct_of_occurrences)]
#        yield rank, [rank, sum(red_percentage_of_occurrences)]
    ''' End: Methods to get distribution from top-k hashtags
    '''
    
    
    ''' Start: Methods to get entropy and focus for all hashtags
    '''
    def map_hashtag_object_to_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak(self, hashtag, hashtag_object):
        mf_lid_to_occurrence_count = get_mf_lid_to_occurrence_count(hashtag_object)
        points = [ getLocationFromLid(lid.replace('_', ' ')) for lid,_ in hashtag_object['ltuo_lid_and_s_interval']]
        # Determine peak
        ltuo_iid_and_tuo_interval_and_occurrence_count = get_ltuo_iid_and_tuo_interval_and_occurrence_count(hashtag_object)
        peak_tuo_iid_and_tuo_interval_and_occurrence_count = \
            max(ltuo_iid_and_tuo_interval_and_occurrence_count, key=lambda (_, (__, occurrence_count)): occurrence_count)
        peak_iid = peak_tuo_iid_and_tuo_interval_and_occurrence_count[0]
        yield hashtag_object['hashtag'], [hashtag_object['hashtag'], 
                                          len(hashtag_object['ltuo_lid_and_s_interval']), 
                                          entropy(mf_lid_to_occurrence_count, False), 
                                          focus(mf_lid_to_occurrence_count), 
                                          getRadiusOfGyration(points),
                                          peak_iid]
    ''' End: Methods to get entropy and focus for all hashtags
    '''
        
    ''' Start: Methods to get stats related to intervals
        interval_stats = [peak, percentage_of_occurrences, cumulative_percentage_of_occurrences, entropy, focus, coverage]
    '''
    def map_hashtag_object_to_tuo_iid_and_interval_stats(self, hashtag, hashtag_object):
        ltuo_iid_and_tuo_interval_and_occurrence_count = \
            get_ltuo_iid_and_tuo_interval_and_occurrence_count(hashtag_object)
        # Peak data
        peak_tuo_iid_and_tuo_interval_and_occurrence_count = \
            max(ltuo_iid_and_tuo_interval_and_occurrence_count, key=lambda (_, (__, occurrence_count)): occurrence_count)
        peak_iid = peak_tuo_iid_and_tuo_interval_and_occurrence_count[0]
        # Points for entropy, focus and coverage
#        mf_lid_to_occurrence_count = get_mf_lid_to_occurrence_count(hashtag_object)
#        points = [ getLocationFromLid(lid.replace('_', ' ')) for lid,_ in hashtag_object['ltuo_lid_and_s_interval']]
        # Occurrence percentage and cumulative occurrence percentage
        current_val = 0.0
        total_occurrences = sum(data[1][1] for data in ltuo_iid_and_tuo_interval_and_occurrence_count)
        for iid, (_, occurrence_count) in ltuo_iid_and_tuo_interval_and_occurrence_count:
            is_peak = 0.0
            if iid==peak_iid: is_peak=1.0
            current_val+=occurrence_count
#            yield iid, [is_peak, occurrence_count/total_occurrences, current_val/total_occurrences, entropy(mf_lid_to_occurrence_count, False), focus(mf_lid_to_occurrence_count), getRadiusOfGyration(points)]
            yield iid, [is_peak, occurrence_count/total_occurrences, current_val/total_occurrences]
    def red_tuo_iid_and_ito_interval_stats_to_tuo_iid_and_reduced_interval_stats(self, iid, ito_interval_stats):
        total_is_peaks = 0.0
        red_percentage_of_occurrences = []
        red_cumulative_percentage_of_occurrences = []
#        red_cumulative_entropy = []
#        red_cumulative_focus = []
#        red_cumulative_coverage = []
#        for (is_peak, percentage_of_occurrences, cumulative_percentage_of_occurrences, entropy, focus, coverage)  in\
        for (is_peak, percentage_of_occurrences, cumulative_percentage_of_occurrences)  in\
                ito_interval_stats:
            total_is_peaks+=is_peak
            red_percentage_of_occurrences.append(percentage_of_occurrences)
            red_cumulative_percentage_of_occurrences.append(cumulative_percentage_of_occurrences)
#            red_cumulative_entropy.append(entropy)
#            red_cumulative_focus.append(focus)
#            red_cumulative_coverage.append(coverage)
        yield iid, [iid, [total_is_peaks, 
                          np.mean(red_percentage_of_occurrences), 
                          np.mean(red_cumulative_percentage_of_occurrences), 
#                          np.mean(red_cumulative_entropy), 
#                          np.mean(red_cumulative_focus), 
#                          np.mean(red_cumulative_coverage)
                    ]]
    ''' End: Methods to get stats related to intervals
    '''
        
    ''' Start: Methods to get iid and perct of occurrence difference
    '''
    def map_hashtag_object_to_tuo_iid_and_perct_of_occurrence_difference(self, hashtag, hashtag_object):
        ltuo_iid_and_tuo_interval_and_occurrence_count = \
            get_ltuo_iid_and_tuo_interval_and_occurrence_count(hashtag_object)
        total_occurrences = sum(data[1][1] for data in ltuo_iid_and_tuo_interval_and_occurrence_count)+0.
        peak_tuo_iid_and_tuo_interval_and_occurrence_count = \
            max(ltuo_iid_and_tuo_interval_and_occurrence_count, key=lambda (_, (__, occurrence_count)): occurrence_count)
        peak_iid = peak_tuo_iid_and_tuo_interval_and_occurrence_count[0]
        if peak_iid<288:
            previous_count = 0.0
            for iid, (_, occurrence_count) in ltuo_iid_and_tuo_interval_and_occurrence_count:
                change = occurrence_count-previous_count
                previous_count = occurrence_count+0.
                yield iid, change/total_occurrences
    def red_tuo_iid_and_ito_perct_of_occurrence_difference_to_tuo_iid_and_mean_perct_of_occurrence_difference(self, iid, ito_perct_of_occurrence_difference):
        red_perct_of_occurrence_difference = []
        for perct_of_occurrence_difference in \
                ito_perct_of_occurrence_difference:
            red_perct_of_occurrence_difference.append(perct_of_occurrence_difference)
        yield iid, [iid, np.mean(red_perct_of_occurrence_difference)]
    ''' End: Methods to get iid and perct of occurrence difference
    '''
        
        
    
    ''' Start: Methods to get stats related to normalized intervals
        interval_stats = [percentage_of_occurrences, entropy, focus, coverage,
                            distance_from_overall_entropy, 
                            distance_from_overall_focus,
                            distance_from_overall_coverage ]
    '''
    def map_hashtag_object_to_tuo_norm_iid_and_interval_stats(self, hashtag, hashtag_object):
        def distance_from_overall_locality_stat(overall_stat, current_stat): return overall_stat-current_stat
        ltuo_iid_and_tuo_interval_and_lids = \
            get_ltuo_iid_and_tuo_interval_and_lids(hashtag_object)
        peak_tuo_iid_and_tuo_interval_and_lids = \
            max(ltuo_iid_and_tuo_interval_and_lids, key=lambda (_, (__, lids)): len(lids))
        peak_iid = peak_tuo_iid_and_tuo_interval_and_lids[0]
#        total_occurrences = sum(len(data[1][1]) for data in peak_tuo_iid_and_tuo_interval_and_lids)
        # Overall locality stats
        overall_mf_lid_to_occurrence_count = get_mf_lid_to_occurrence_count(hashtag_object)
        overall_points = [ getLocationFromLid(lid.replace('_', ' ')) for lid,_ in hashtag_object['ltuo_lid_and_s_interval']]
        overall_entropy = entropy(overall_mf_lid_to_occurrence_count, False)
        overall_focus = focus(overall_mf_lid_to_occurrence_count)[1]
        overall_coverage = getRadiusOfGyration(overall_points)
        total_occurrences = sum(len(lids) for (iid, (interval, lids)) in ltuo_iid_and_tuo_interval_and_lids)
        for iid, (_, lids) in ltuo_iid_and_tuo_interval_and_lids:
            mf_lid_to_occurrence_count = defaultdict(float)
            for lid in lids: mf_lid_to_occurrence_count[lid]+=1
            points = [getLocationFromLid(lid.replace('_', ' ')) for lid in lids]
            
            current_entropy = entropy(mf_lid_to_occurrence_count, False)
            current_focus = focus(mf_lid_to_occurrence_count)[1]
            current_coverage = getRadiusOfGyration(points)
            
            yield iid-peak_iid, [len(lids)/total_occurrences, current_entropy, current_focus, current_coverage, 
                                    distance_from_overall_locality_stat(overall_entropy, current_entropy),
                                    distance_from_overall_locality_stat(overall_focus, current_focus),
                                    distance_from_overall_locality_stat(overall_coverage, current_coverage),]
#            yield '%s_%s'%(iid-peak_iid, peak_iid), [len(lids)/total_occurrences, entropy(mf_lid_to_occurrence_count, False), focus(mf_lid_to_occurrence_count)[1], getRadiusOfGyration(points)]
    def red_tuo_norm_iid_and_ito_interval_stats_to_tuo_norm_iid_and_reduced_interval_stats(self, norm_iid, ito_interval_stats):
        red_percentage_of_occurrences = []
        red_cumulative_entropy = []
        red_cumulative_focus = []
        red_cumulative_coverage = []
        red_distance_from_overall_entropy = []
        red_distance_from_overall_focus = []
        red_distance_from_overall_coverage = []
        for (percentage_of_occurrences, entropy, focus, coverage, 
                    distance_from_overall_entropy, distance_from_overall_focus, distance_from_overall_coverage)  in\
                ito_interval_stats:
            red_percentage_of_occurrences.append(percentage_of_occurrences)
            red_cumulative_entropy.append(entropy)
            red_cumulative_focus.append(focus)
            red_cumulative_coverage.append(coverage)
            red_distance_from_overall_entropy.append(distance_from_overall_entropy)
            red_distance_from_overall_focus.append(distance_from_overall_focus)
            red_distance_from_overall_coverage.append(distance_from_overall_coverage)
        yield norm_iid, [norm_iid, 
                         [ 
                          np.mean(red_percentage_of_occurrences), 
                          np.mean(red_cumulative_entropy), 
                          np.mean(red_cumulative_focus), 
                          np.mean(red_cumulative_coverage),
                          np.mean(red_distance_from_overall_entropy), 
                          np.mean(red_distance_from_overall_focus), 
                          np.mean(red_distance_from_overall_coverage),
                    ]]
    ''' End: Methods to get stats related to intervals
    '''
    
    ''' Start: Methods to temporal distance between hashtags
    '''
    def map_hashtag_object_to_tuo_lid_other_lid_and_temporal_distance(self, hashtag, hashtag_object):
        # Get peak
        ltuo_iid_and_tuo_interval_and_lids = \
            get_ltuo_iid_and_tuo_interval_and_lids(hashtag_object)
        peak_tuo_iid_and_tuo_interval_and_lids = \
            max(ltuo_iid_and_tuo_interval_and_lids, key=lambda (_, (__, lids)): len(lids))
        peak_iid = peak_tuo_iid_and_tuo_interval_and_lids[0]
        # Get valid intervals with corresponding focus lids
        ltuo_valid_iid_and_focus_lid = []
        ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count = \
            get_ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count(hashtag_object)
        so_observed_focus_lids = set()
        for iid, (interval, ltuo_lid_and_occurrence_count) in \
                ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count:
            if (iid-peak_iid) in VALID_IID_RANGE:
                focus_lid  = focus(dict(ltuo_lid_and_occurrence_count))[0]
                if focus_lid not in so_observed_focus_lids:
                    ltuo_valid_iid_and_focus_lid.append([iid, focus_lid])
                    so_observed_focus_lids.add(focus_lid)
        for (valid_iid1, focus_lid1), (valid_iid2, focus_lid2) in combinations(ltuo_valid_iid_and_focus_lid, 2):
            yield ':ilab:'.join(sorted([focus_lid1, focus_lid2])), abs(valid_iid1-valid_iid2)
#            yield focus_lid1, [focus_lid2, valid_iid1-valid_iid2]
#            yield focus_lid2, [focus_lid1, valid_iid2-valid_iid1]
    def red_tuo_lid_other_lid_and_ito_temporal_distance_to_tuo_lid_other_lid_and_temporal_ditance(self, lid_other_lid, ito_temporal_distance):
#        red_mf_other_lid_to_temporal_distances = defaultdict(list)
#        for other_lid, temporal_distacne in \
#                ito_other_lid_and_temporal_distance:
#            red_mf_other_lid_to_temporal_distances[other_lid].append(temporal_distacne)
#        # Filter other lids that haven't been observed minimum number of times and
#        # yield the mean distance for others.
#        for other_lid in red_mf_other_lid_to_temporal_distances.keys()[:]:
#            if len(red_mf_other_lid_to_temporal_distances[other_lid])<MIN_NUMBER_OF_SHARED_HASHTAGS:
#                    del red_mf_other_lid_to_temporal_distances[other_lid]
#            else: red_mf_other_lid_to_temporal_distances[other_lid]=np.mean(red_mf_other_lid_to_temporal_distances[other_lid])
#        if red_mf_other_lid_to_temporal_distances:
#            yield lid, [lid, red_mf_other_lid_to_temporal_distances]
        red_temporal_distances = []
        for temporal_distance in ito_temporal_distance: red_temporal_distances.append(temporal_distance)
        if len(red_temporal_distances)>=MIN_NUMBER_OF_SHARED_HASHTAGS:
            yield lid_other_lid, [lid_other_lid, np.mean(red_temporal_distances)]
    ''' End: Methods to temporal distance between hashtags    
    '''
            
    ''' Start: Methods to get lid oc-occurrences
    '''
    def map_hashtag_object_to_tuo_lid_other_lid_and_one(self, hashtag, hashtag_object):
        # Get peak
        ltuo_iid_and_tuo_interval_and_lids = \
            get_ltuo_iid_and_tuo_interval_and_lids(hashtag_object)
        peak_tuo_iid_and_tuo_interval_and_lids = \
            max(ltuo_iid_and_tuo_interval_and_lids, key=lambda (_, (__, lids)): len(lids))
        peak_iid = peak_tuo_iid_and_tuo_interval_and_lids[0]
        # Get valid intervals with corresponding focus lids
        ltuo_valid_iid_and_focus_lid = []
        ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count = \
            get_ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count(hashtag_object)
        so_observed_focus_lids = set()
        for iid, (interval, ltuo_lid_and_occurrence_count) in \
                ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count:
            if (iid-peak_iid) in VALID_IID_RANGE: 
                focus_lid  = focus(dict(ltuo_lid_and_occurrence_count))[0]
                if focus_lid not in so_observed_focus_lids:
                    ltuo_valid_iid_and_focus_lid.append([iid, focus_lid])
                    so_observed_focus_lids.add(focus_lid)
        for (valid_iid1, focus_lid1), (valid_iid2, focus_lid2) in combinations(ltuo_valid_iid_and_focus_lid, 2):
            yield ':ilab:'.join(sorted([focus_lid1, focus_lid2])), 1.0
#            yield focus_lid2, [focus_lid1, valid_iid2-valid_iid1]
    def red_tuo_lid_other_lid_and_ito_ane_to_lid_other_lid_and_cooccurrence_count(self, lid_other_lid, ito_one):
        red_total_co_occurrences = 0.0
        for count in ito_one: red_total_co_occurrences+=count
        # Filter other lids that haven't been observed minimum number of times
        if red_total_co_occurrences >= MIN_NUMBER_OF_SHARED_HASHTAGS:
            yield lid_other_lid, [lid_other_lid, red_total_co_occurrences]
    ''' End: Methods to temporal distance between hashtags    
    '''
    
    ''' Start: Methods to get distribution of peak lids
    '''
    def map_hashtag_object_to_tuo_no_of_peak_lids_and_count(self, hashtag, hashtag_object):
        ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count = \
            get_ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count(hashtag_object)
        so_observed_focus_lids = set()
        for _, (_, ltuo_lid_and_occurrence_count) in \
                ltuo_iid_and_tuo_interval_and_ltuo_lid_and_occurrence_count:
            focus_lid  = focus(dict(ltuo_lid_and_occurrence_count))[0]
            if focus_lid not in so_observed_focus_lids:
                so_observed_focus_lids.add(focus_lid)
        yield len(so_observed_focus_lids), 1.
    def red_tuo_no_of_peak_lids_and_ito_count_to_no_of_peak_lids_and_count(self, no_of_peak_lids, ito_count):
        red_count = []
        for count in ito_count: red_count.append(count)
        yield no_of_peak_lids, [no_of_peak_lids, sum(red_count)]
    ''' End: Methods to get distribution of peak lids   
    '''
    
    ''' Start: Methods to get hashtags for a lid
    '''
    def map_hashtag_object_to_tuo_lid_and_hashtags(self, hashtag, hashtag_object):
        so_observed_focus_lids = get_so_observed_focus_lids(hashtag_object)
        for lid in so_observed_focus_lids: yield lid, hashtag_object['hashtag']
    def red_tuo_lid_and_ito_hashtags_to_tuo_lid_and_hashtags(self, lid, ito_hashtags):
        yield lid, [lid, list(set(list(ito_hashtags)))]
    ''' End: Methods to get hashtags for a lid
    '''
    
         
    ''' MR Jobs
    '''
    def job_load_hashtag_object(self): return [
                                               self.mr(
                                                       mapper=self.map_checkin_line_to_tuo_hashtag_and_ltuo_lid_and_occurrence_time, 
                                                       mapper_final=self.mapf_checkin_line_to_tuo_hashtag_and_ltuo_lid_and_occurrence_time, 
                                                       reducer=self.red_tuo_hashtag_and_ito_ltuo_lid_and_occurrence_time_to_tuo_hashtag_and_hashtag_object
                                                       )
                                               ]
    def job_load_preprocessed_hashtag_object(self): return [
                                               self.mr(
                                                       mapper=self.map_hashtag_object_string_to_tuo_of_hashtag_and_hashtag_object, 
                                                       reducer=self.red_tuo_hashtag_and_ito_hashtag_object_to_tuo_hashtag_and_hashtag_object
                                                       )
                                               ]
    def job_write_tuo_normalized_occurrence_count_and_distribution_value(self): 
        return [
                   self.mr(
                           mapper=self.map_checkin_line_to_tuo_hashtag_and_occurrence_count, 
                           mapper_final=self.mapf_checkin_line_to_tuo_hashtag_and_occurrence_count, 
                           reducer=self.red_tuo_hashtag_and_ito_occurrence_count_to_tuo_normalized_occurrence_count_and_one
                           )
                   ] + \
                   [
                    self.mr(
                           mapper=self.emptyMapper, 
                           reducer=self.red_tuo_normalized_occurrence_count_and_ito_one_to_tuo_normalized_occurrence_count_and_distribution_value
                           )
                    ]
    def job_write_tweet_count_stats(self):
        return [
                   self.mr(
                           mapper=self.map_checkin_line_to_tuo_stat_and_stat_value, 
                           mapper_final=self.mapf_checkin_line_to_tuo_stat_and_stat_value, 
                           reducer=self.red_tuo_stat_and_ito_stat_value_to_tuo_stat_and_stat_value
                           )
                   ]
    def job_tuo_no_of_hashtags_and_count(self):
        return self.job_load_hashtag_object() + \
               [
                            self.mr(
                                   mapper=self.map_hashtag_object_to_tuo_no_of_hashtags_and_count, 
                                   reducer=self.red_tuo_no_of_hashtags_and_ito_count_to_no_of_hashtags_and_count
                                   )
                       ]
    def job_tuo_no_of_locations_and_count(self):
        return self.job_load_hashtag_object() + \
               [
                            self.mr(
                                   mapper=self.map_hashtag_object_to_tuo_no_of_locations_and_count, 
                                   reducer=self.red_tuo_no_of_locations_and_ito_count_to_no_of_locations_and_count
                                   )
                       ]
    def job_write_tuo_lid_and_distribution_value(self):
        return [
                   self.mr(
                           mapper=self.map_checkin_line_to_tuo_lid_and_occurrence_count, 
                           mapper_final=self.mapf_checkin_line_to_tuo_lid_and_occurrence_count, 
                           reducer=self.red_tuo_lid_and_ito_occurrence_count_to_tuo_lid_and_occurrence_count
                           )
                   ]
    def job_write_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak(self):
        return self.job_load_hashtag_object() + \
                [
                    self.mr(
                           mapper=self.map_hashtag_object_to_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak, 
                           )
                   ]
    def job_write_tuo_rank_and_average_percentage_of_occurrences(self):
        return self.job_load_preprocessed_hashtag_object() + \
                [
                    self.mr(
                           mapper=self.map_hashtag_object_to_tuo_cum_prct_of_occurrences_and_prct_of_occurrences, 
                           reducer=self.red_tuo_rank_and_ito_percentage_of_occurrences_to_tuo_rank_and_avg_cum_prct_of_occurrences_and_avg_prct_of_occurrences
                           )
                   ]
    def job_write_tuo_iid_and_interval_stats(self):
        return self.job_load_hashtag_object() + \
                [
                        self.mr(
                               mapper=self.map_hashtag_object_to_tuo_iid_and_interval_stats, 
                               reducer=self.red_tuo_iid_and_ito_interval_stats_to_tuo_iid_and_reduced_interval_stats
                               )
                   ]
    def job_write_tuo_iid_and_perct_of_occurrence_difference(self):
        return self.job_load_preprocessed_hashtag_object() + \
                [
                        self.mr(
                               mapper=self.map_hashtag_object_to_tuo_iid_and_perct_of_occurrence_difference, 
                               reducer=self.red_tuo_iid_and_ito_perct_of_occurrence_difference_to_tuo_iid_and_mean_perct_of_occurrence_difference
                               )
                   ]
    def job_write_tuo_norm_iid_and_interval_stats(self):
        return self.job_load_hashtag_object() + \
                [
                        self.mr(
                               mapper=self.map_hashtag_object_to_tuo_norm_iid_and_interval_stats, 
                               reducer=self.red_tuo_norm_iid_and_ito_interval_stats_to_tuo_norm_iid_and_reduced_interval_stats
                               )
                   ]
    def job_write_tuo_lid_and_ltuo_other_lid_and_temporal_distance(self):
        return self.job_load_preprocessed_hashtag_object() + \
               [
                            self.mr(
                                   mapper=self.map_hashtag_object_to_tuo_lid_other_lid_and_temporal_distance, 
                                   reducer=self.red_tuo_lid_other_lid_and_ito_temporal_distance_to_tuo_lid_other_lid_and_temporal_ditance
                                   )
                       ]
    def job_write_tuo_lid_and_ltuo_other_lid_and_no_of_co_occurrences(self):
        return self.job_load_preprocessed_hashtag_object() + \
               [
                            self.mr(
                                   mapper=self.map_hashtag_object_to_tuo_lid_other_lid_and_one, 
                                   reducer=self.red_tuo_lid_other_lid_and_ito_ane_to_lid_other_lid_and_cooccurrence_count
                                   )
                       ]
    def job_tuo_high_accuracy_lid_and_distribution(self):
        return [
                   self.mr(
                           mapper=self.map_checkin_line_to_tuo_high_accuracy_lid_and_ltuo_lid_and_count, 
                           mapper_final=self.mapf_checkin_line_to_tuo_high_accuracy_lid_and_ltuo_lid_and_count, 
                           reducer=self.red_tuo_lid_and_ito_count_and_occurrence_time_to_tuo_lid_and_count
                           )
                   ]
    def job_write_tuo_no_of_peak_lids_and_count(self):
        return self.job_load_preprocessed_hashtag_object() + \
               [
                            self.mr(
                                   mapper=self.map_hashtag_object_to_tuo_no_of_peak_lids_and_count, 
                                   reducer=self.red_tuo_no_of_peak_lids_and_ito_count_to_no_of_peak_lids_and_count
                                   )
                       ]
    def job_write_tuo_lid_and_hashtags(self):
        return self.job_load_preprocessed_hashtag_object() + \
               [
                            self.mr(
                                   mapper=self.map_hashtag_object_to_tuo_lid_and_hashtags, 
                                   reducer=self.red_tuo_lid_and_ito_hashtags_to_tuo_lid_and_hashtags
                                   )
                       ]
    def steps(self):
        pass
#        return self.job_load_hashtag_object()
#        return self.job_tuo_high_accuracy_lid_and_distribution()
        return self.job_tuo_no_of_hashtags_and_count()
#        return self.job_tuo_no_of_locations_and_count()
#        return self.job_load_preprocessed_hashtag_object()
#        return self.job_write_tuo_normalized_occurrence_count_and_distribution_value()
#        return self.job_write_tweet_count_stats()
#        return self.job_write_tuo_lid_and_distribution_value()
#        return self.job_write_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak()
#        return self.job_write_tuo_rank_and_average_percentage_of_occurrences()
#        return self.job_write_tuo_iid_and_interval_stats()
#        return self.job_write_tuo_iid_and_perct_of_occurrence_difference()
#        return self.job_write_tuo_norm_iid_and_interval_stats()
#        return self.job_write_tuo_lid_and_ltuo_other_lid_and_temporal_distance()
#        return self.job_write_tuo_lid_and_ltuo_other_lid_and_no_of_co_occurrences()
#        return self.job_write_tuo_no_of_peak_lids_and_count()
#        return self.job_write_tuo_lid_and_hashtags()
if __name__ == '__main__':
    MRAnalysis.run()
