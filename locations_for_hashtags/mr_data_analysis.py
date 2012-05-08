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
MIN_HASHTAG_OCCURENCES = 100

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
    ltuo_lid_and_occurrence_time = []
    for ltuo_lid_and_occurrence_time in ito_ltuo_lid_and_occurrence_time: 
        for tuo_lid_and_occurrence_time in ltuo_lid_and_occurrence_time: 
            ltuo_lid_and_occurrence_time.append(tuo_lid_and_occurrence_time)
    if ltuo_lid_and_occurrence_time:
        if len(ltuo_lid_and_occurrence_time)>=MIN_HASHTAG_OCCURENCES:
            return {
                    'hashtag': hashtag, 
                    'ltuo_lid_and_s_occurrence_time': sorted(ltuo_lid_and_occurrence_time, key=lambda t: t[1])
                    }

class MRDataAnalysis(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(MRDataAnalysis, self).__init__(*args, **kwargs)
        self.mf_hashtag_to_ltuo_lid_and_occurrence_time = defaultdict(list)
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
    def map_hashtag_object_to_tuo_hashtag_and_occurrences_count(self, hashtag, hashtag_object):
        yield hashtag, [hashtag, len(hashtag_object['ltuo_lid_and_s_occurrence_time'])]
            
    ''' MR Jobs
    '''
    def job_load_hashtag_object(self): return [
                                               self.mr(
                                                       mapper=self.map_checkin_line_to_tuo_hashtag_and_ltuo_lid_and_occurrence_time, 
                                                       mapper_final=self.mapf_checkin_line_to_tuo_hashtag_and_ltuo_lid_and_occurrence_time, 
                                                       reducer=self.red_tuo_hashtag_and_ito_ltuo_lid_and_occurrence_time_to_tuo_hashtag_and_hashtag_object
                                                       )
                                               ]
    def job_write_tuo_hashtag_and_occurrences_count(self): 
        return self.job_load_hashtag_object() +\
                [self.mr( mapper=self.map_hashtag_object_to_tuo_hashtag_and_occurrences_count, )]
    
    def steps(self):
        pass
#        return self.job_load_hashtag_object()
        return self.job_write_tuo_hashtag_and_occurrences_count()
if __name__ == '__main__':
    MRDataAnalysis.run()
