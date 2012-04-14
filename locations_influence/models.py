'''
Created on Apr 13, 2012

@author: krishnakamath
'''
from library.classes import GeneralMethods
from library.file_io import FileIO
import numpy as np
from operator import itemgetter
from settings import tuo_location_and_tuo_neighbor_location_and_pure_influence_score_file, \
    train_location_objects_file  

def iterateJsonFromFile(file):
    for data in FileIO.iterateJsonFromFile(file):
        if 'PARAMS_DICT' not in data: yield data

class InfluenceMeasuringModels(object):
    ID_FIRST_OCCURRENCE = 'first_occurrence'
    ID_FIRST_AND_LAST_OCCURRENCE = 'first_and_last_occurrence'
    ID_AGGREGATE_OCCURRENCE = 'aggregate_occurrence'
    ID_WEIGHTED_AGGREGATE_OCCURRENCE = 'weighted_aggregate_occurrence'
    @staticmethod
    def first_occurrence(location_occurrences, neighbor_location_occurrences):
        first_location_occurrence=sorted(location_occurrences)[0]
        first_neighbor_location_occurrence=sorted(neighbor_location_occurrences)[0]
        if first_location_occurrence>first_neighbor_location_occurrence: return 1.0
        elif first_location_occurrence<first_neighbor_location_occurrence: return -1.0
        else: return 0.0
    @staticmethod
    def first_and_last_occurrence(location_occurrences, neighbor_location_occurrences):
        pass
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
                                                  (InfluenceMeasuringModels.ID_FIRST_AND_LAST_OCCURRENCE, InfluenceMeasuringModels.first_and_last_occurrence),
                                                  (InfluenceMeasuringModels.ID_AGGREGATE_OCCURRENCE, InfluenceMeasuringModels.aggregate_occurrence),
                                                  (InfluenceMeasuringModels.ID_WEIGHTED_AGGREGATE_OCCURRENCE, InfluenceMeasuringModels.weighted_aggregate_occurrence),
                                                  ])

class Experiments(object):
    @staticmethod
    def generate_influence_scores():
        models_ids = [
                      InfluenceMeasuringModels.ID_FIRST_OCCURRENCE, InfluenceMeasuringModels.ID_FIRST_AND_LAST_OCCURRENCE, 
                      InfluenceMeasuringModels.ID_AGGREGATE_OCCURRENCE, InfluenceMeasuringModels.ID_WEIGHTED_AGGREGATE_OCCURRENCE,
                  ]
        for model_id in models_ids:
            GeneralMethods.runCommand('rm -rf %s'%tuo_location_and_tuo_neighbor_location_and_pure_influence_score_file)
            for line_count, location_object in enumerate(iterateJsonFromFile(train_location_objects_file)):
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
                    for hashtag in location_hashtag_set.difference(neighbor_location_hashtag_set): pure_influence_scores.append(1.0)
                    for hashtag in neighbor_location_hashtag_set.difference(location_hashtag_set): pure_influence_scores.append(-1.0)
                    mean_pure_influence_score = np.mean(pure_influence_scores)
                    tuo_neighbor_location_and_pure_influence_score.append([neighbor_location, mean_pure_influence_score])
                tuo_neighbor_location_and_pure_influence_score = sorted(tuo_neighbor_location_and_pure_influence_score, key=itemgetter(1))
                FileIO.writeToFileAsJson([location_object['id'], tuo_neighbor_location_and_pure_influence_score], 
                                         tuo_neighbor_location_and_pure_influence_score)
        
    @staticmethod
    def run():
        Experiments.generate_influence_scores()

Experiments.run()
        
    
    