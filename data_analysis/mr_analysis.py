'''
Created on May 7, 2012

@author: krishnakamath
'''
import cjson, time
from library.mrjobwrapper import ModifiedMRJob
from library.twitter import getDateTimeObjectFromTweetTimestamp
from library.geo import getLatticeLid
from library.classes import GeneralMethods
from collections import defaultdict
from datetime import datetime
from library.stats import entropy, focus
from operator import itemgetter
import numpy as np

#LOCATION_ACCURACY = 1.45 # 100 miles
LOCATION_ACCURACY = 0.001 # 100 miles
TIME_UNIT_IN_SECONDS = 60*10 # 10 minutes
MIN_HASHTAG_OCCURENCES = 10
#START_TIME, END_TIME = datetime(2011, 3, 1), datetime(2012, 3, 31)
START_TIME, END_TIME = datetime(2011, 3, 1), datetime(2011, 4, 28)


#Distribution Paramter
DISTRIBUTION_ACCURACY = 100

# Top K rank analysis
K_TOP_RANK = 100

# Parameters for the MR Job that will be logged.
HASHTAG_STARTING_WINDOW, HASHTAG_ENDING_WINDOW = time.mktime(START_TIME.timetuple()), time.mktime(END_TIME.timetuple())
PARAMS_DICT = dict(PARAMS_DICT = True,
                   LOCATION_ACCURACY=LOCATION_ACCURACY,
                   MIN_HASHTAG_OCCURENCES=MIN_HASHTAG_OCCURENCES,
                   TIME_UNIT_IN_SECONDS = TIME_UNIT_IN_SECONDS,
                   HASHTAG_STARTING_WINDOW = HASHTAG_STARTING_WINDOW, HASHTAG_ENDING_WINDOW = HASHTAG_ENDING_WINDOW,
                   K_TOP_RANK = K_TOP_RANK,
                   )

def iterate_hashtag_occurrences(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    lid = getLatticeLid(l, LOCATION_ACCURACY)
    for h in data['h']: yield h.lower(), [lid, GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS)]
    
def combine_hashtag_instances(hashtag, ito_ltuo_lid_and_occurrence_time):
    combined_ltuo_lid_and_occurrence_time = []
    for ltuo_lid_and_occurrence_time in ito_ltuo_lid_and_occurrence_time: 
        for tuo_lid_and_occurrence_time in ltuo_lid_and_occurrence_time: 
            combined_ltuo_lid_and_occurrence_time.append(tuo_lid_and_occurrence_time)
    if combined_ltuo_lid_and_occurrence_time:
#        e, l = min(combined_ltuo_lid_and_occurrence_time, key=lambda t: t[1]), max(combined_ltuo_lid_and_occurrence_time, key=lambda t: t[1])
#        if len(combined_ltuo_lid_and_occurrence_time)>=MIN_HASHTAG_OCCURENCES and \
#                e[1]>=HASHTAG_STARTING_WINDOW and l[1]<=HASHTAG_ENDING_WINDOW:
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
    def map_hashtag_object_to_tuo_rank_and_percentage_of_occurrences(self, hashtag, hashtag_object):
        mf_lid_to_occurrence_count = get_mf_lid_to_occurrence_count(hashtag_object)
        yield len(mf_lid_to_occurrence_count), 1
#        obj = get_mf_interval_to_mf_lid_to_occurrence_count(hashtag_object)
#        yield hashtag, mf_lid_to_occurrence_count
#        ltuo_lid_and_r_occurrence_count = sorted(mf_lid_to_occurrence_count.items(), key=itemgetter(1), reverse=True)
#        total_occurrence_count = float(sum(zip(*ltuo_lid_and_r_occurrence_count)[1]))
#        for rank, (_, occurrence_count) in enumerate(ltuo_lid_and_r_occurrence_count[:K_TOP_RANK]):
#            yield rank+1, occurrence_count/total_occurrence_count
    def red_tuo_rank_and_ito_percentage_of_occurrences_to_tuo_rank_and_average_percentage_of_occurrences(self, rank, ito_percentage_of_occurrences):
        red_percentage_of_occurrences = []
        for percentage_of_occurrence in ito_percentage_of_occurrences: red_percentage_of_occurrences.append(percentage_of_occurrence)
#        yield rank, [rank, np.mean(red_percentage_of_occurrences)]
        yield rank, [rank, sum(red_percentage_of_occurrences)]
    ''' End: Methods to get distribution from top-k hashtags
    '''
    
    
    ''' Start: Methods to get entropy and focus for all hashtags
    '''
    def map_hashtag_object_to_tuo_hashtag_and_occurrence_count_and_entropy_and_focus(self, hashtag, hashtag_object):
        mf_lid_to_occurrence_count = get_mf_lid_to_occurrence_count(hashtag_object)
        yield hashtag_object['hashtag'], [hashtag_object['hashtag'], len(hashtag_object['ltuo_lid_and_s_interval']), entropy(mf_lid_to_occurrence_count, False), focus(mf_lid_to_occurrence_count)]
    ''' End: Methods to get entropy and focus for all hashtags
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
    def job_write_tuo_lid_and_distribution_value(self):
        return [
                   self.mr(
                           mapper=self.map_checkin_line_to_tuo_lid_and_occurrence_count, 
                           mapper_final=self.mapf_checkin_line_to_tuo_lid_and_occurrence_count, 
                           reducer=self.red_tuo_lid_and_ito_occurrence_count_to_tuo_lid_and_occurrence_count
                           )
                   ]
    def job_write_tuo_hashtag_and_occurrence_count_and_entropy_and_focus(self):
        return self.job_load_hashtag_object() + \
                [
                    self.mr(
                           mapper=self.map_hashtag_object_to_tuo_hashtag_and_occurrence_count_and_entropy_and_focus, 
                           )
                   ]
    def job_write_tuo_rank_and_average_percentage_of_occurrences(self):
        return self.job_load_hashtag_object() + \
                [
                    self.mr(
                           mapper=self.map_hashtag_object_to_tuo_rank_and_percentage_of_occurrences, 
                           reducer=self.red_tuo_rank_and_ito_percentage_of_occurrences_to_tuo_rank_and_average_percentage_of_occurrences
                           )
                   ]
    def steps(self):
        pass
#        return self.job_load_hashtag_object()
#        return self.job_write_tuo_normalized_occurrence_count_and_distribution_value()
#        return self.job_write_tweet_count_stats()
#        return self.job_write_tuo_lid_and_distribution_value()
#        return self.job_write_tuo_hashtag_and_occurrence_count_and_entropy_and_focus()
        return self.job_write_tuo_rank_and_average_percentage_of_occurrences()
    
if __name__ == '__main__':
    MRAnalysis.run()
