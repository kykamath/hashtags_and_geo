'''
Created on Oct 3, 2012

@author: kykamath
'''
from collections import defaultdict
from datetime import datetime
from library.geo import UTMConverter
from library.mrjobwrapper import ModifiedMRJob
from library.nlp import getWordsFromRawEnglishMessage
from library.twitter import getDateTimeObjectFromTweetTimestamp
from operator import itemgetter
import cjson
import nltk
import time

ACCURACY = 10**4 # UTM boxes in sq.m

# Minimum number of hashtag occurrences
# Used by HashtagsExtractor
MIN_HASHTAG_OCCURRENCES = 50

# Start time for data analysis
START_TIME, END_TIME = datetime(2011, 3, 1), datetime(2012, 7, 31)

# Parameters for the MR Job that will be logged.
HASHTAG_STARTING_WINDOW = time.mktime(START_TIME.timetuple())
HASHTAG_ENDING_WINDOW = time.mktime(END_TIME.timetuple())

PARAMS_DICT = dict(
                   PARAMS_DICT = True,
                   ACCURACY = ACCURACY,
                   MIN_HASHTAG_OCCURRENCES = MIN_HASHTAG_OCCURRENCES,
                   HASHTAG_STARTING_WINDOW = HASHTAG_STARTING_WINDOW,
                   HASHTAG_ENDING_WINDOW = HASHTAG_ENDING_WINDOW,
                   )

def iterate_word_hashtag_pairs(line):
    data = cjson.decode(line)
    if data['h']:
        l = None
        if 'geo' in data: l = data['geo']
        else: l = data['bb']
        words = filter(
                           lambda w: w[0]!='#',
                           getWordsFromRawEnglishMessage(data['tx'])
                       )
        words = filter(lambda (w, pos): pos=='NN' or pos=='NP', nltk.pos_tag(words))
        words = map(itemgetter(0), words)
        
        t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
        for w in words:
            for h in data['h']: yield w, h.lower(), l, t
            
class HashtagsExtractor(ModifiedMRJob):
    '''
    hashtag_object = {
                      'hashtag' : hashtag,
                      'ltuo_occ_time_and_word': ltuo_occ_time_and_word
                      'ltuo_occ_time_and_occ_location': [],
                      'num_of_occurrences' : 0
                      'num_of_words' : 0
                    }
    '''
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self,  min_hashtag_occurrences = MIN_HASHTAG_OCCURRENCES, *args, **kwargs):
        super(HashtagsExtractor, self).__init__(*args, **kwargs)
        self.min_hashtag_occurrences = min_hashtag_occurrences
        self.mf_hastag_to_ltuo_occ_time_and_occ_location = defaultdict(list)
        self.mf_hastag_to_ltuo_occ_time_and_word = defaultdict(list)
    def mapper(self, key, line):
        if False: yield # I'm a generator!
        for word, hashtag, location, occ_time in iterate_word_hashtag_pairs(line):
#            location = UTMConverter.getUTMIdInLatLongFormFromLatLong( location[0], location[1], accuracy=ACCURACY)
            self.mf_hastag_to_ltuo_occ_time_and_occ_location[hashtag].append((occ_time, location))
            self.mf_hastag_to_ltuo_occ_time_and_word[hashtag].append((occ_time, word))
    def mapper_final(self):
        for hashtag, ltuo_occ_time_and_occ_location in self.mf_hastag_to_ltuo_occ_time_and_occ_location.iteritems():
            hashtag_object = {
                              'hashtag': hashtag,
                              'ltuo_occ_time_and_occ_location': ltuo_occ_time_and_occ_location,
                              'ltuo_occ_time_and_word': self.mf_hastag_to_ltuo_occ_time_and_word[hashtag]
                              }
            yield hashtag, hashtag_object
    def _get_combined_hashtag_object(self, hashtag, hashtag_objects):
        combined_hashtag_object = {'hashtag': hashtag, 'ltuo_occ_time_and_occ_location': [], 'words': []}
        for hashtag_object in hashtag_objects:
            combined_hashtag_object['ltuo_occ_time_and_occ_location']+=hashtag_object['ltuo_occ_time_and_occ_location']
            combined_hashtag_object['ltuo_occ_time_and_word']+=hashtag_object['ltuo_occ_time_and_word']
        combined_hashtag_object['num_of_occurrences'] = len(combined_hashtag_object['ltuo_occ_time_and_occ_location']) 
        combined_hashtag_object['num_of_words'] = len(combined_hashtag_object['ltuo_occ_time_and_word']) 
        return combined_hashtag_object
    def reducer(self, hashtag, hashtag_objects):
        combined_hashtag_object = self._get_combined_hashtag_object(hashtag, hashtag_objects)
        e = min(combined_hashtag_object['ltuo_occ_time_and_occ_location'], key=lambda t: t[0])
        l = max(combined_hashtag_object['ltuo_occ_time_and_occ_location'], key=lambda t: t[0])
#        if combined_hashtag_object['num_of_occurrences'] >= \
#                self.min_hashtag_occurrences and \
#                e[0]>=HASHTAG_STARTING_WINDOW and l[0]<=HASHTAG_ENDING_WINDOW:
        combined_hashtag_object['ltuo_occ_time_and_occ_location'] = \
            sorted(combined_hashtag_object['ltuo_occ_time_and_occ_location'], key=itemgetter(0))
        combined_hashtag_object['ltuo_occ_time_and_word'] = \
            sorted(combined_hashtag_object['ltuo_occ_time_and_word'], key=itemgetter(0))
        yield hashtag, combined_hashtag_object

#class HashtagsExtractor(ModifiedMRJob):
#    '''
#    word_object = {
#                      'word': word,
#                      'mf_hastag_to_ltuo_occ_time_and_occ_location' : mf_hastag_to_ltuo_occ_time_and_occ_location
#                    }
#    '''
#    DEFAULT_INPUT_PROTOCOL='raw_value'
#    def __init__(self,  min_hashtag_occurrences = MIN_HASHTAG_OCCURRENCES, *args, **kwargs):
#        super(HashtagsExtractor, self).__init__(*args, **kwargs)
#        self.mf_word_to_mf_hastag_to_ltuo_occ_time_and_occ_location = {}
#    def mapper(self, key, line):
#        if False: yield # I'm a generator!
#        for word, hashtag, (location, occ_time) in iterate_word_hashtag_pairs(line):
#            if word not in self.mf_word_to_mf_hastag_to_ltuo_occ_time_and_occ_location:
#                self.mf_word_to_mf_hastag_to_ltuo_occ_time_and_occ_location[word] = defaultdict(list)
#            location = UTMConverter.getUTMIdInLatLongFormFromLatLong( location[0], location[1], accuracy=ACCURACY)
#            self.mf_word_to_mf_hastag_to_ltuo_occ_time_and_occ_location[word][hashtag].append((occ_time, location))
#    def mapper_final(self):
#        for word, mf_hastag_to_ltuo_occ_time_and_occ_location in\
#                    self.mf_word_to_mf_hastag_to_ltuo_occ_time_and_occ_location.iteritems():
#            word_object = {
#                           'word': word,
#                           'mf_hastag_to_ltuo_occ_time_and_occ_location': mf_hastag_to_ltuo_occ_time_and_occ_location
#                        }
#            yield word, word_object
#    def _get_combined_hashtag_object(self, hashtag, hashtag_objects):
#        combined_hashtag_object = {'hashtag': hashtag, 'ltuo_occ_time_and_occ_location': []}
#        for hashtag_object in hashtag_objects:
#            combined_hashtag_object['ltuo_occ_time_and_occ_location']+=hashtag_object['ltuo_occ_time_and_occ_location']
#        combined_hashtag_object['num_of_occurrences'] = len(combined_hashtag_object['ltuo_occ_time_and_occ_location']) 
#        return combined_hashtag_object
#    def reducer(self, hashtag, hashtag_objects):
#        combined_hashtag_object = self._get_combined_hashtag_object(hashtag, hashtag_objects)
#        e = min(combined_hashtag_object['ltuo_occ_time_and_occ_location'], key=lambda t: t[0])
#        l = max(combined_hashtag_object['ltuo_occ_time_and_occ_location'], key=lambda t: t[0])
#        if combined_hashtag_object['num_of_occurrences'] >= \
#                self.min_hashtag_occurrences and \
#                e[0]>=HASHTAG_STARTING_WINDOW and l[0]<=HASHTAG_ENDING_WINDOW:
#            combined_hashtag_object['ltuo_occ_time_and_occ_location'] = \
#                sorted(combined_hashtag_object['ltuo_occ_time_and_occ_location'], key=itemgetter(0))
#            yield hashtag, combined_hashtag_object

if __name__ == '__main__':
    HashtagsExtractor.run()