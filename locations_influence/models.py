'''
Created on Apr 13, 2012

@author: krishnakamath
'''
from library.classes import GeneralMethods
from library.file_io import FileIO
from collections import defaultdict
import numpy as np
from operator import itemgetter
from settings import tuo_location_and_tuo_neighbor_location_and_pure_influence_score_file, \
    location_objects_file, tuo_location_and_tuo_neighbor_location_and_influence_score_file
from analysis import iterateJsonFromFile
from mr_analysis import START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER

class InfluenceMeasuringModels(object):
    ID_FIRST_OCCURRENCE = 'first_occurrence'
    ID_MEAN_OCCURRENCE = 'mean_occurrence'
    ID_AGGREGATE_OCCURRENCE = 'aggregate_occurrence'
    ID_WEIGHTED_AGGREGATE_OCCURRENCE = 'weighted_aggregate_occurrence'
    @staticmethod
    def first_occurrence(location_occurrences, neighbor_location_occurrences):
        first_location_occurrence=sorted(location_occurrences)[0]
        first_neighbor_location_occurrence=sorted(neighbor_location_occurrences)[0]
        if first_location_occurrence<first_neighbor_location_occurrence: return 1.0
        elif first_location_occurrence>first_neighbor_location_occurrence: return -1.0
        else: return 0.0
    @staticmethod
    def mean_occurrence(location_occurrences, neighbor_location_occurrences):
        mean_location_occurrence=np.mean(location_occurrences)
        mean_neighbor_location_occurrence=np.mean(neighbor_location_occurrences)
        if mean_location_occurrence<mean_neighbor_location_occurrence: return 1.0
        elif mean_location_occurrence>mean_neighbor_location_occurrence: return -1.0
        else: return 0.0
    @staticmethod
    def _get_occurrences_stats(occurrences1, occurrences2):
        no_of_occurrences_after_appearing_in_location, no_of_occurrences_before_appearing_in_location = 0., 0.
        occurrences1=sorted(occurrences1)
        occurrences2=sorted(occurrences2)
        no_of_total_occurrences_between_location_pair = len(occurrences1)*len(occurrences2)*1.
        for occurrence1 in occurrences1:
            for occurrence2 in occurrences2:
                if occurrence1<occurrence2: no_of_occurrences_after_appearing_in_location+=1
                elif occurrence1>occurrence2: no_of_occurrences_before_appearing_in_location+=1
        return no_of_occurrences_after_appearing_in_location, no_of_occurrences_before_appearing_in_location, no_of_total_occurrences_between_location_pair
    @staticmethod
    def aggregate_occurrence(location_occurrences, neighbor_location_occurrences):
        (no_of_occurrences_after_appearing_in_location, \
         no_of_occurrences_before_appearing_in_location, \
         no_of_total_occurrences_between_location_pair)= InfluenceMeasuringModels._get_occurrences_stats(location_occurrences, neighbor_location_occurrences)
        return (no_of_occurrences_after_appearing_in_location - no_of_occurrences_before_appearing_in_location) / no_of_total_occurrences_between_location_pair
    @staticmethod
    def weighted_aggregate_occurrence(location_occurrences, neighbor_location_occurrences):
        (no_of_occurrences_after_appearing_in_location, \
         no_of_occurrences_before_appearing_in_location, \
         no_of_total_occurrences_between_location_pair) =\
            InfluenceMeasuringModels._get_occurrences_stats(location_occurrences, neighbor_location_occurrences)
        total_nof_occurrences = float(len(location_occurrences) + len(neighbor_location_occurrences))
        ratio_of_occurrences_in_location = len(location_occurrences)/total_nof_occurrences
        ratio_of_occurrences_in_neighbor_location = len(neighbor_location_occurrences)/total_nof_occurrences
        return (
                ratio_of_occurrences_in_location*no_of_occurrences_after_appearing_in_location \
                - ratio_of_occurrences_in_neighbor_location*no_of_occurrences_before_appearing_in_location
                ) / no_of_total_occurrences_between_location_pair
MF_INFLUENCE_MEASURING_MODELS_TO_MODEL_ID = dict([
                                                  (InfluenceMeasuringModels.ID_FIRST_OCCURRENCE, InfluenceMeasuringModels.first_occurrence),
                                                  (InfluenceMeasuringModels.ID_MEAN_OCCURRENCE, InfluenceMeasuringModels.mean_occurrence),
                                                  (InfluenceMeasuringModels.ID_AGGREGATE_OCCURRENCE, InfluenceMeasuringModels.aggregate_occurrence),
                                                  (InfluenceMeasuringModels.ID_WEIGHTED_AGGREGATE_OCCURRENCE, InfluenceMeasuringModels.weighted_aggregate_occurrence),
                                                  ])

class Experiments(object):
    @staticmethod
    def generate_tuo_location_and_tuo_neighbor_location_and_pure_influence_score(models_ids, startTime, endTime, outputFolder):
        for model_id in models_ids:
            output_file = tuo_location_and_tuo_neighbor_location_and_pure_influence_score_file%model_id
            GeneralMethods.runCommand('rm -rf %s'%output_file)
            for line_count, location_object in enumerate(iterateJsonFromFile(
                     location_objects_file%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
                     )):
                print line_count, model_id
                tuo_neighbor_location_and_pure_influence_score = []
                location_hashtag_set = set(location_object['hashtags'])
                for neighbor_location, mf_hashtag_to_tuo_occurrences_and_time_range in location_object['links'].iteritems():
                    pure_influence_scores = []
                    for hashtag, (neighbor_location_occurrences, time_range) in mf_hashtag_to_tuo_occurrences_and_time_range.iteritems():
                        if hashtag in location_object['hashtags']:
                            location_occurrences = location_object['hashtags'][hashtag][0]
                            pure_influence_scores.append(MF_INFLUENCE_MEASURING_MODELS_TO_MODEL_ID[model_id](location_occurrences, neighbor_location_occurrences))
                    neighbor_location_hashtag_set = set(mf_hashtag_to_tuo_occurrences_and_time_range.keys())
#                    for hashtag in location_hashtag_set.difference(neighbor_location_hashtag_set): pure_influence_scores.append(1.0)
#                    for hashtag in neighbor_location_hashtag_set.difference(location_hashtag_set): pure_influence_scores.append(-1.0)
                    mean_pure_influence_score = np.mean(pure_influence_scores)
                    tuo_neighbor_location_and_pure_influence_score.append([neighbor_location, mean_pure_influence_score])
                tuo_neighbor_location_and_pure_influence_score = sorted(tuo_neighbor_location_and_pure_influence_score, key=itemgetter(1))
                FileIO.writeToFileAsJson([location_object['id'], tuo_neighbor_location_and_pure_influence_score], output_file)
    @staticmethod
    def load_tuo_location_and_tuo_neighbor_location_and_pure_influence_score(model_id):
        return [(location, tuo_neighbor_location_and_pure_influence_score)
                 for location, tuo_neighbor_location_and_pure_influence_score in 
                 iterateJsonFromFile(tuo_location_and_tuo_neighbor_location_and_pure_influence_score_file%model_id)]
    @staticmethod
    def generate_tuo_location_and_tuo_neighbor_location_and_influence_score(models_ids, startTime, endTime, outputFolder):
        def get_hashtag_weights(map_from_hashtag_to_tuples_of_occurrences_and_time_range):
            total_occurrences = sum([len(occurrences) 
                                     for hashtag, (occurrences, time_range) in 
                                     map_from_hashtag_to_tuples_of_occurrences_and_time_range.iteritems()]) + 0.
            return dict([(hashtag, len(occurrences)/total_occurrences)
                for hashtag, (occurrences, time_range) in 
                map_from_hashtag_to_tuples_of_occurrences_and_time_range.iteritems()])
        def get_location_weights(hashtags_for_source_location, map_from_location_to_hashtags):
            set_of_hashtags_for_source_location = set(hashtags_for_source_location.keys())
            return dict([(location, len(set(hashtags.keys()).intersection(set_of_hashtags_for_source_location))/(len(set_of_hashtags_for_source_location)+0.))
                         for location, hashtags in 
                         map_from_location_to_hashtags.iteritems()])
        for model_id in models_ids:
            output_file = tuo_location_and_tuo_neighbor_location_and_influence_score_file%model_id
            GeneralMethods.runCommand('rm -rf %s'%output_file)
            for line_count, location_object in enumerate(iterateJsonFromFile(
                     location_objects_file%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
                     )):
                print line_count, model_id
                tuo_neighbor_location_and_influence_score = []
                mf_hashtag_to_hashtag_weights = get_hashtag_weights(location_object['hashtags'])
                mf_location_to_location_weights = get_location_weights(location_object['hashtags'], location_object['links'])
                for neighbor_location, map_from_hashtag_to_tuples_of_occurrences_and_time_range in location_object['links'].iteritems():
                    influence_scores = []
                    for hashtag, (neighbor_location_occurrences, time_range) in map_from_hashtag_to_tuples_of_occurrences_and_time_range.iteritems():
                        if hashtag in location_object['hashtags']:
                            location_occurrences = location_object['hashtags'][hashtag][0]
                            pure_influence_score = MF_INFLUENCE_MEASURING_MODELS_TO_MODEL_ID[model_id](location_occurrences, neighbor_location_occurrences)
                            influence_scores.append(mf_hashtag_to_hashtag_weights[hashtag]*pure_influence_score)
#                    for hashtag in location_hashtag_set.difference(neighbor_location_hashtag_set): pure_influence_scores.append(1.0)
#                    for hashtag in neighbor_location_hashtag_set.difference(location_hashtag_set): pure_influence_scores.append(-1.0)
                    mean_influence_scores = np.mean(influence_scores)
                    tuo_neighbor_location_and_influence_score.append([neighbor_location, 
                                                                       mf_location_to_location_weights[neighbor_location]*mean_influence_scores])
                tuo_neighbor_location_and_influence_score = sorted(tuo_neighbor_location_and_influence_score, key=itemgetter(1))
                FileIO.writeToFileAsJson([location_object['id'], tuo_neighbor_location_and_influence_score], output_file)
    @staticmethod
    def load_tuo_location_and_tuo_neighbor_location_and_influence_score(model_id):
        return [(location, tuo_neighbor_location_and_influence_score)
                 for location, tuo_neighbor_location_and_influence_score in 
                 iterateJsonFromFile(tuo_location_and_tuo_neighbor_location_and_influence_score_file%model_id)]
    @staticmethod
    def load_tuo_location_and_tuo_neighbor_location_and_locations_influence_score(model_id, noOfInfluencers=None):
        '''
        noOfInfluencers (k) = The top-k influencers for a location
        '''
        tuo_location_and_tuo_neighbor_location_and_influence_score = Experiments.load_tuo_location_and_tuo_neighbor_location_and_influence_score(model_id)
        mf_location_to_tuo_neighbor_location_and_locations_influence_score = defaultdict(list)
        for neighbor_location, tuo_location_and_influence_score in tuo_location_and_tuo_neighbor_location_and_influence_score:
            if not noOfInfluencers: tuo_location_and_influence_score = filter(lambda (location, influence_score): influence_score<0, tuo_location_and_influence_score)
            else: tuo_location_and_influence_score = filter(lambda (location, influence_score): influence_score<0, tuo_location_and_influence_score)[:noOfInfluencers]
            for location, influence_score in tuo_location_and_influence_score:
                mf_location_to_tuo_neighbor_location_and_locations_influence_score[location].append([neighbor_location, abs(influence_score)])
        for location in mf_location_to_tuo_neighbor_location_and_locations_influence_score.keys()[:]:
            tuo_neighbor_location_and_locations_influence_score = mf_location_to_tuo_neighbor_location_and_locations_influence_score[location]
            mf_location_to_tuo_neighbor_location_and_locations_influence_score[location] = sorted(tuo_neighbor_location_and_locations_influence_score, key=itemgetter(1), reverse=True)
        return mf_location_to_tuo_neighbor_location_and_locations_influence_score.items()

    @staticmethod
    def run():
        models_ids = [
#                      InfluenceMeasuringModels.ID_FIRST_OCCURRENCE, 
                      InfluenceMeasuringModels.ID_MEAN_OCCURRENCE, 
#                      InfluenceMeasuringModels.ID_AGGREGATE_OCCURRENCE, 
#                      InfluenceMeasuringModels.ID_WEIGHTED_AGGREGATE_OCCURRENCE,
                  ]
#        Experiments.generate_tuo_location_and_tuo_neighbor_location_and_pure_influence_score(models_ids, START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER)
        Experiments.generate_tuo_location_and_tuo_neighbor_location_and_influence_score(models_ids, START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER)

if __name__ == '__main__':
    Experiments.run()
        
    
    