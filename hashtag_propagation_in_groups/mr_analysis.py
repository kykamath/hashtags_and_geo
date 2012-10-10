'''
Created on Oct 3, 2012

@author: kykamath
'''
from collections import defaultdict
from datetime import datetime
from itertools import chain, groupby
from library.geo import UTMConverter
from library.mrjobwrapper import ModifiedMRJob
from library.nlp import getWordsFromRawEnglishMessage
from library.twitter import getDateTimeObjectFromTweetTimestamp
from operator import itemgetter
from scipy import stats
import cjson
import networkx as nx
import nltk
import time

ACCURACY = 10**4 # UTM boxes in sq.m

# Minimum number of hashtag occurrences
# Used by HashtagsExtractor
MIN_HASHTAG_OCCURRENCES = 10

# Start time for data analysis
START_TIME, END_TIME = datetime(2011, 3, 1), datetime(2012, 7, 31)

# Parameters for the MR Job that will be logged.
HASHTAG_STARTING_WINDOW = time.mktime(START_TIME.timetuple())
HASHTAG_ENDING_WINDOW = time.mktime(END_TIME.timetuple())

# Minimum number of word occurrences expected for each hashtag to be considered a valid word
MIN_WORD_OCCURRENCES_PER_HASHTAG = 10

# Significance threshold for p-value of association measure tests
SIGNIFICANCE_THRESHOLD = 0.05

## Temporal Width of a hashtag group
#TEMPORAL_WIDTH_OF_HASHTAG_GROUP_IN_SECONDS = 24*60*10

PARAMS_DICT = dict(
                   PARAMS_DICT = True,
                   ACCURACY = ACCURACY,
                   MIN_HASHTAG_OCCURRENCES = MIN_HASHTAG_OCCURRENCES,
                   HASHTAG_STARTING_WINDOW = HASHTAG_STARTING_WINDOW,
                   HASHTAG_ENDING_WINDOW = HASHTAG_ENDING_WINDOW,
                   MIN_WORD_OCCURRENCES_PER_HASHTAG = MIN_WORD_OCCURRENCES_PER_HASHTAG,
                   SIGNIFICANCE_THRESHOLD = SIGNIFICANCE_THRESHOLD
                   )

def iterate_hashtag_with_words(line):
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
        for h in data['h']: yield h.lower(), words, l, t
            
class HashtagsExtractor(ModifiedMRJob):
    '''
    hashtag_object = {
                      'hashtag' : hashtag,
                      'ltuo_occ_time_and_words': ltuo_occ_time_and_words
                      'ltuo_occ_time_and_occ_location': [],
                      'num_of_occurrences' : 0,
                      'valid_words': valid_words # Added in WordObjectExtractor
                    }
    '''
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self,  min_hashtag_occurrences = MIN_HASHTAG_OCCURRENCES, *args, **kwargs):
        super(HashtagsExtractor, self).__init__(*args, **kwargs)
        self.min_hashtag_occurrences = min_hashtag_occurrences
        self.mf_hastag_to_ltuo_occ_time_and_occ_location = defaultdict(list)
        self.mf_hastag_to_ltuo_occ_time_and_words = defaultdict(list)
    def mapper(self, key, line):
        if False: yield # I'm a generator!
        for hashtag, words, location, occ_time in iterate_hashtag_with_words(line):
#            location = UTMConverter.getUTMIdInLatLongFormFromLatLong( location[0], location[1], accuracy=ACCURACY)
            self.mf_hastag_to_ltuo_occ_time_and_occ_location[hashtag].append((occ_time, location))
            self.mf_hastag_to_ltuo_occ_time_and_words[hashtag].append((occ_time, words))
    def mapper_final(self):
        for hashtag, ltuo_occ_time_and_occ_location in self.mf_hastag_to_ltuo_occ_time_and_occ_location.iteritems():
            hashtag_object = {
                              'hashtag': hashtag,
                              'ltuo_occ_time_and_occ_location': ltuo_occ_time_and_occ_location,
                              'ltuo_occ_time_and_words': self.mf_hastag_to_ltuo_occ_time_and_words[hashtag]
                              }
            yield hashtag, hashtag_object
    def _get_combined_hashtag_object(self, hashtag, hashtag_objects):
        combined_hashtag_object = {
                                       'hashtag': hashtag,
                                       'ltuo_occ_time_and_occ_location': [],
                                       'ltuo_occ_time_and_words': []
                                   }
        for hashtag_object in hashtag_objects:
            combined_hashtag_object['ltuo_occ_time_and_occ_location']+=hashtag_object['ltuo_occ_time_and_occ_location']
            combined_hashtag_object['ltuo_occ_time_and_words']+=hashtag_object['ltuo_occ_time_and_words']
        combined_hashtag_object['num_of_occurrences'] = len(combined_hashtag_object['ltuo_occ_time_and_occ_location']) 
        return combined_hashtag_object
    def reducer(self, hashtag, hashtag_objects):
        combined_hashtag_object = self._get_combined_hashtag_object(hashtag, hashtag_objects)
        e = min(combined_hashtag_object['ltuo_occ_time_and_occ_location'], key=lambda t: t[0])
        l = max(combined_hashtag_object['ltuo_occ_time_and_occ_location'], key=lambda t: t[0])
        if combined_hashtag_object['num_of_occurrences'] >= \
                self.min_hashtag_occurrences and \
                e[0]>=HASHTAG_STARTING_WINDOW and l[0]<=HASHTAG_ENDING_WINDOW:
            combined_hashtag_object['ltuo_occ_time_and_occ_location'] = \
                sorted(combined_hashtag_object['ltuo_occ_time_and_occ_location'], key=itemgetter(0))
            combined_hashtag_object['ltuo_occ_time_and_words'] = \
                sorted(combined_hashtag_object['ltuo_occ_time_and_words'], key=itemgetter(0))
            yield hashtag, combined_hashtag_object

class WordObjectExtractor(ModifiedMRJob):
    '''
    contingency_table_object = {
                                'word': word,
                                'hashtag': hashtag,
                                'n00': Number of occurrences of word and hashtag,
                                'n01': Number of occurrences of word without hashtag,
                                'n10': Number of occurrences of hashtag without word,
                                'n11': Number of occurrences without hashtag and without word,
                                }
    word_object = {
                    'word': word,
                    'total_word_occurrences': total_word_occurrences,
                    'contingency_table_objects': contingency_table_objects
                }
    '''
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(WordObjectExtractor, self).__init__(*args, **kwargs)
        self.mf_word_to_mf_hashtag_to_ltuo_word_and_occ_time_and_occ_location = defaultdict(dict)
    def _get_words_above_min_threshold(self, ltuo_occ_time_and_words):
        l_words = map(itemgetter(1), ltuo_occ_time_and_words)
        words = sorted(list(chain(*l_words)))
        ltuo_word_and_count = [(word, len(list(it_word))) for word, it_word in groupby(words)]
        ltuo_word_and_count = filter(
                                     lambda (w,c): c>MIN_WORD_OCCURRENCES_PER_HASHTAG,
                                     ltuo_word_and_count
                                 )
        return map(itemgetter(0), ltuo_word_and_count)
        
    def mapper(self, key, line):
        if False: yield # I'm a generator!
        data = cjson.decode(line)
        if 'hashtag' in data:
            hashtag = data['hashtag']
            ltuo_occ_time_and_occ_location = data['ltuo_occ_time_and_occ_location']
            ltuo_occ_time_and_words = data['ltuo_occ_time_and_words']
            valid_words = self._get_words_above_min_threshold(ltuo_occ_time_and_words)
            ltuo_word_and_occ_time_and_occ_location = []
            for (occ_time, occ_location),(_, words) in zip(ltuo_occ_time_and_occ_location, ltuo_occ_time_and_words):
                for word in filter(lambda w: w in valid_words, words):
                    ltuo_word_and_occ_time_and_occ_location.append([word, occ_time, occ_location])
            for word in valid_words:
                self.mf_word_to_mf_hashtag_to_ltuo_word_and_occ_time_and_occ_location[word][hashtag]=\
                                                                                ltuo_word_and_occ_time_and_occ_location
    def mapper_final(self):
        for word, mf_hashtag_to_ltuo_word_and_occ_time_and_occ_location in\
                self.mf_word_to_mf_hashtag_to_ltuo_word_and_occ_time_and_occ_location.iteritems():
            ltuo_hashtag_and_ltuo_word_and_occ_time_and_occ_location =\
                                                        mf_hashtag_to_ltuo_word_and_occ_time_and_occ_location.items()
            yield word, ltuo_hashtag_and_ltuo_word_and_occ_time_and_occ_location
    def reducer(self, word, it_ltuo_hashtag_and_ltuo_word_and_occ_time_and_occ_location):
        so_hashtag = set()
        mf_hashtag_to_n00 = defaultdict(float)
        mf_hashtag_to_n10 = defaultdict(float)
        total_word_occurrences = 0.0
        for ltuo_hashtag_and_ltuo_word_and_occ_time_and_occ_location in\
                it_ltuo_hashtag_and_ltuo_word_and_occ_time_and_occ_location:
            for hashtag, ltuo_word_and_occ_time_and_occ_location in\
                    ltuo_hashtag_and_ltuo_word_and_occ_time_and_occ_location:
                so_hashtag.add(hashtag)
                for neigbhor_word, occ_time, occ_location in ltuo_word_and_occ_time_and_occ_location:
                    if neigbhor_word==word: 
                        total_word_occurrences+=1
                        mf_hashtag_to_n00[hashtag]+=1
                    else: mf_hashtag_to_n10[hashtag]+=1
        contingency_table_objects = []
        for hashtag in so_hashtag:
            contingency_table_object = { 'word': word, 'hashtag': hashtag}
            contingency_table_object['n00'] = mf_hashtag_to_n00.get(hashtag, 0.0)
            contingency_table_object['n10'] = mf_hashtag_to_n10.get(hashtag, 0.0)
            contingency_table_object['n01'] = total_word_occurrences - contingency_table_object['n00']
            contingency_table_objects.append(contingency_table_object)
        word_object = {
                    'word': word,
                    'total_word_occurrences': total_word_occurrences,
                    'contingency_table_objects': contingency_table_objects
                }
        yield word, word_object

class WordHashtagContingencyTableObjectExtractor(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(WordHashtagContingencyTableObjectExtractor, self).__init__(*args, **kwargs) 
        self.word_object_extractor = WordObjectExtractor()
    def mapper(self, key, word_object): yield '', [word_object]
    def reducer(self, empty_key, it_word_objects):
        word_objects = list(chain(*it_word_objects))
        total_words = sum(map(itemgetter('total_word_occurrences'), word_objects))
        for word_object in word_objects:
            for contingency_table_object in word_object['contingency_table_objects']:
                total_occurrences_of_either_word_or_hashtag =\
                        contingency_table_object['n00']+contingency_table_object['n01']+contingency_table_object['n10']
                contingency_table_object['n11'] = total_words - total_occurrences_of_either_word_or_hashtag 
                yield (
                           '%s_%s'%(contingency_table_object['word'], contingency_table_object['hashtag']),
                           contingency_table_object
                       )
    def steps(self):
        return self.word_object_extractor.steps() + [self.mr(mapper=self.mapper, reducer=self.reducer)]

class AbstractAssociatioMeasure(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(AbstractAssociatioMeasure, self).__init__(*args, **kwargs)
        self.word_hashtag_contingency_table_object_extractor = WordHashtagContingencyTableObjectExtractor()
    def association_measure_p_value(self, contingency_table_object):
        raise NotImplementedError
    def get_contingency_array(self, contingency_table_object):
        return [
                    [contingency_table_object['n00'], contingency_table_object['n01']],
                    [contingency_table_object['n10'], contingency_table_object['n11']]
                ]
    def mapper(self, key, contingency_table_object):
        measure_score, pvalue = self.association_measure_stats(contingency_table_object)
        if pvalue<SIGNIFICANCE_THRESHOLD:
            yield '', {'word': contingency_table_object['word'], 'hashtag': '#'+contingency_table_object['hashtag']}
    def reducer(self, empty_key, values):
        graph = nx.Graph()
        for value in values: graph.add_edge(value['word'], value['hashtag'])
        connected_components = nx.connected_components(graph)
        ltuo_num_of_hashtags_and_connected_component = map(
                                                               lambda c: (
                                                                              len(filter(lambda w: w[0]=='#', c)),
                                                                              sorted(c)
                                                                          ),
                                                               connected_components
                                                           )
        ltuo_num_of_hashtags_and_connected_component.sort(key=itemgetter(0), reverse=True)
        for num_of_hashtags, connected_component in ltuo_num_of_hashtags_and_connected_component:
            yield num_of_hashtags, [num_of_hashtags, connected_component]
    def steps(self):
        return self.word_hashtag_contingency_table_object_extractor.steps() +\
                [self.mr(mapper=self.mapper, reducer=self.reducer)]
#    def steps(self):
#        return self.word_hashtag_contingency_table_object_extractor.steps() +\
#                [self.mr(mapper=self.mapper)]
    
class DemoAssociatioMeasure(AbstractAssociatioMeasure):
    def __init__(self, *args, **kwargs):
        super(DemoAssociatioMeasure, self).__init__(*args, **kwargs)
    def association_measure_stats(self, contingency_table_object):
        return (1.0, 0.0)

class FisherExactTest(AbstractAssociatioMeasure):
    def __init__(self, *args, **kwargs):
        super(FisherExactTest, self).__init__(*args, **kwargs)
    def association_measure_stats(self, contingency_table_object):
        return stats.fisher_exact(self.get_contingency_array(contingency_table_object))
      
if __name__ == '__main__':
#    HashtagsExtractor.run()
#    WordObjectExtractor.run()
#    WordHashtagContingencyTableObjectExtractor.run()
#    DemoAssociatioMeasure.run()
    FisherExactTest.run()
