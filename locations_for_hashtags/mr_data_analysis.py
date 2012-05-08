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

LOCATION_ACCURACY = 1.45 # 100 miles
TIME_UNIT_IN_SECONDS = 60*10 # 10 minutes
MIN_HASHTAG_OCCURENCES = 0

#Distribution Paramter
DISTRIBUTION_ACCURACY = 100

# Parameters for the MR Job that will be logged.
PARAMS_DICT = dict(PARAMS_DICT = True,
                   LOCATION_ACCURACY=LOCATION_ACCURACY,
                   MIN_HASHTAG_OCCURENCES=MIN_HASHTAG_OCCURENCES,
                   TIME_UNIT_IN_SECONDS = TIME_UNIT_IN_SECONDS,
                   )

def iterate_hashtag_occurrences(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    lattice_lid = getLatticeLid(l, LOCATION_ACCURACY)
    for h in data['h']: yield h.lower(), [lattice_lid, GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS)]
    
def combine_hashtag_instances(hashtag, ito_ltuo_lid_and_occurrence_time):
    combined_ltuo_lid_and_occurrence_time = []
    for ltuo_lid_and_occurrence_time in ito_ltuo_lid_and_occurrence_time: 
        for tuo_lid_and_occurrence_time in ltuo_lid_and_occurrence_time: 
            combined_ltuo_lid_and_occurrence_time.append(tuo_lid_and_occurrence_time)
    if combined_ltuo_lid_and_occurrence_time:
        if len(combined_ltuo_lid_and_occurrence_time)>=MIN_HASHTAG_OCCURENCES:
            return {
                    'hashtag': hashtag, 
                    'ltuo_lid_and_s_occurrence_time': sorted(combined_ltuo_lid_and_occurrence_time, key=lambda t: t[1])
                    }

class MRDataAnalysis(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(MRDataAnalysis, self).__init__(*args, **kwargs)
        self.mf_hashtag_to_ltuo_lid_and_occurrence_time = defaultdict(list)
        self.mf_hashtag_to_occurrence_count = defaultdict(float)
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
    
    def steps(self):
        pass
#        return self.job_load_hashtag_object()
        return self.job_write_tuo_normalized_occurrence_count_and_distribution_value()
if __name__ == '__main__':
    MRDataAnalysis.run()
