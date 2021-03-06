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
    location_objects_file, tuo_location_and_tuo_neighbor_location_and_influence_score_file, \
    tuo_location_and_tuo_neighbor_location_and_mf_influence_type_and_similarity_file, \
    tuo_location_and_tuo_neighbor_location_and_sharing_affinity_score_file, \
    w_extra_hashtags_tag, wout_extra_hashtags_tag, f_hashtag_objects, \
    f_ltuo_location_and_ltuo_hashtag_and_occurrence_time, \
    f_ltuo_hashtag_and_ltuo_location_and_pure_influence_score
from analysis import iterateJsonFromFile
from itertools import groupby
from datetime import datetime
from mr_analysis import LOCATION_ACCURACY
from library.geo import isWithinBoundingBox, getLocationFromLid, getLatticeLid
from sklearn.cluster.affinity_propagation_ import AffinityPropagation

JACCARD_SIMILARITY = 'jaccard_similarity'
class InfluenceMeasuringModels(object):
    ID_FIRST_OCCURRENCE = 'first_occurrence'
    ID_MEAN_OCCURRENCE = 'mean_occurrence'
    ID_AGGREGATE_OCCURRENCE = 'aggregate_occurrence'
    ID_WEIGHTED_AGGREGATE_OCCURRENCE = 'weighted_aggregate_occurrence'
    TYPE_COMPLETE_INFLUENCE = 'complete_influence'
    TYPE_OUTGOING_INFLUENCE = 'outgoing_influence'
    TYPE_INCOMING_INFLUENCE = 'incoming_influence'
    INFLUENCE_PROPERTIES = {
                            TYPE_OUTGOING_INFLUENCE: {'label':  'Influenced locations similarity', 'color': 'r', 'marker': '*', 'id': 'influencing_vector'},
                            TYPE_INCOMING_INFLUENCE: {'label': 'Influencing locations similarity', 'color': 'b', 'marker': 's', 'id': 'influenced_by_vector'},
                            TYPE_COMPLETE_INFLUENCE: {'label': 'Influence', 'color': 'g', 'marker': 'o', 'id': 'influence_vector'},
                            JACCARD_SIMILARITY: {'label': 'Jaccard', 'color': 'm', 'marker': 's', 'id': 'jaccard_similarity'},
                            }
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
    SINGLETON_ZONE_ID = 'singleton_zone'
    @staticmethod
    def generate_tuo_location_and_tuo_neighbor_location_and_pure_influence_score(models_ids, startTime, endTime, outputFolder, hashtag_tag):
        for model_id in models_ids:
#            if w_extra_hashtags: output_file = tuo_location_and_tuo_neighbor_location_and_pure_influence_score_file%(model_id, hashtag_tag)
#            else: output_file = tuo_location_and_tuo_neighbor_location_and_pure_influence_score_file%(model_id, wout_extra_hashtags_tag)
            output_file = tuo_location_and_tuo_neighbor_location_and_pure_influence_score_file%(model_id, hashtag_tag)
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
                    if hashtag_tag==w_extra_hashtags_tag:
                        for hashtag in location_hashtag_set.difference(neighbor_location_hashtag_set): pure_influence_scores.append(1.0)
                        for hashtag in neighbor_location_hashtag_set.difference(location_hashtag_set): pure_influence_scores.append(-1.0)
                    mean_pure_influence_score = np.mean(pure_influence_scores)
                    tuo_neighbor_location_and_pure_influence_score.append([neighbor_location, mean_pure_influence_score])
                tuo_neighbor_location_and_pure_influence_score = sorted(tuo_neighbor_location_and_pure_influence_score, key=itemgetter(1))
                FileIO.writeToFileAsJson([location_object['id'], tuo_neighbor_location_and_pure_influence_score], output_file)
    @staticmethod
    def load_tuo_location_and_tuo_neighbor_location_and_pure_influence_score(model_id, hashtag_tag):
        return [(location, tuo_neighbor_location_and_pure_influence_score)
                 for location, tuo_neighbor_location_and_pure_influence_score in 
                 iterateJsonFromFile(tuo_location_and_tuo_neighbor_location_and_pure_influence_score_file%(model_id, hashtag_tag))]
    @staticmethod
    def generate_tuo_location_and_tuo_neighbor_location_and_influence_score(models_ids, startTime, endTime, outputFolder, hashtag_tag):
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
            output_file = tuo_location_and_tuo_neighbor_location_and_influence_score_file%(model_id, hashtag_tag)
            GeneralMethods.runCommand('rm -rf %s'%output_file)
            for line_count, location_object in enumerate(iterateJsonFromFile(
                     location_objects_file%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
                     )):
                print line_count, model_id
                tuo_neighbor_location_and_influence_score = []
                mf_hashtag_to_hashtag_weights = get_hashtag_weights(location_object['hashtags'])
                mf_location_to_location_weights = get_location_weights(location_object['hashtags'], location_object['links'])
                location_hashtag_set = set(location_object['hashtags'])
                for neighbor_location, mf_hashtag_to_tuo_occurrences_and_time_range in location_object['links'].iteritems():
                    influence_scores = []
                    mf_neighbor_location_hashtag_to_hashtag_weights = get_hashtag_weights(mf_hashtag_to_tuo_occurrences_and_time_range)
                    neighbor_location_hashtag_set = set(mf_hashtag_to_tuo_occurrences_and_time_range.keys())
                    for hashtag, (neighbor_location_occurrences, time_range) in mf_hashtag_to_tuo_occurrences_and_time_range.iteritems():
                        if hashtag in location_object['hashtags']:
                            location_occurrences = location_object['hashtags'][hashtag][0]
                            pure_influence_score = MF_INFLUENCE_MEASURING_MODELS_TO_MODEL_ID[model_id](location_occurrences, neighbor_location_occurrences)
                            influence_scores.append(mf_hashtag_to_hashtag_weights[hashtag]*pure_influence_score)
                    if hashtag_tag==w_extra_hashtags_tag:
                        for hashtag in location_hashtag_set.difference(neighbor_location_hashtag_set): 
                            influence_scores.append(mf_hashtag_to_hashtag_weights[hashtag]*1.0)
    #                        influence_scores.append(1.0)
                        for hashtag in neighbor_location_hashtag_set.difference(location_hashtag_set): 
                            influence_scores.append(mf_neighbor_location_hashtag_to_hashtag_weights[hashtag]*-1.0)
    #                        influence_scores.append(-1.0)
                    mean_influence_scores = np.mean(influence_scores)
                    tuo_neighbor_location_and_influence_score.append([neighbor_location, 
                                                                       mf_location_to_location_weights[neighbor_location]*mean_influence_scores])
                tuo_neighbor_location_and_influence_score = sorted(tuo_neighbor_location_and_influence_score, key=itemgetter(1))
                FileIO.writeToFileAsJson([location_object['id'], tuo_neighbor_location_and_influence_score], output_file)
    @staticmethod
    def load_tuo_location_and_tuo_neighbor_location_and_influence_score(model_id, hashtag_tag):
        return [(location, tuo_neighbor_location_and_influence_score)
                 for location, tuo_neighbor_location_and_influence_score in 
                 iterateJsonFromFile(tuo_location_and_tuo_neighbor_location_and_influence_score_file%(model_id, hashtag_tag))]
    @staticmethod
    def load_tuo_location_and_tuo_neighbor_location_and_locations_influence_score(model_id, hashtag_tag,
                                                                                       noOfInfluencers=None, 
                                                                                       influence_type=InfluenceMeasuringModels.TYPE_INCOMING_INFLUENCE):
        '''
        noOfInfluencers (k) = The top-k influencers for a location
        '''
        tuo_location_and_tuo_neighbor_location_and_influence_score \
            = Experiments.load_tuo_location_and_tuo_neighbor_location_and_influence_score(model_id, hashtag_tag)
        mf_location_to_tuo_neighbor_location_and_locations_influence_score = defaultdict(list)
        for neighbor_location, tuo_location_and_influence_score in tuo_location_and_tuo_neighbor_location_and_influence_score:
            if not noOfInfluencers: 
                if influence_type==InfluenceMeasuringModels.TYPE_INCOMING_INFLUENCE:
                    tuo_location_and_influence_score = filter(lambda (location, influence_score): influence_score<0, tuo_location_and_influence_score)
                else: tuo_location_and_influence_score = filter(lambda (location, influence_score): influence_score>=0, tuo_location_and_influence_score)
            else: 
                if influence_type==InfluenceMeasuringModels.TYPE_INCOMING_INFLUENCE:
                    tuo_location_and_influence_score = filter(lambda (location, influence_score): influence_score<0, tuo_location_and_influence_score)[:noOfInfluencers]
                else: tuo_location_and_influence_score = filter(lambda (location, influence_score): influence_score>0, tuo_location_and_influence_score)[:noOfInfluencers]
            for location, influence_score in tuo_location_and_influence_score:
                mf_location_to_tuo_neighbor_location_and_locations_influence_score[location].append([neighbor_location, abs(influence_score)])
        for location in mf_location_to_tuo_neighbor_location_and_locations_influence_score.keys()[:]:
            tuo_neighbor_location_and_locations_influence_score = mf_location_to_tuo_neighbor_location_and_locations_influence_score[location]
            mf_location_to_tuo_neighbor_location_and_locations_influence_score[location] = sorted(tuo_neighbor_location_and_locations_influence_score, 
                                                                                                       key=itemgetter(1), reverse=True)
        return mf_location_to_tuo_neighbor_location_and_locations_influence_score.items()
    @staticmethod
    def load_tuo_location_and_boundary_influence_score(model_id, hashtag_tag, boundary=[[-90,-180], [90, 180]], noOfInfluencers=None):
        mf_location_to_global_influence_score = {}
        mf_location_to_mf_influence_type_to_influence_score = defaultdict(dict)
        mf_location_to_tuo_neighbor_location_and_locations_influencing_score = \
            dict(Experiments.load_tuo_location_and_tuo_neighbor_location_and_locations_influence_score(model_id, hashtag_tag, noOfInfluencers, InfluenceMeasuringModels.TYPE_INCOMING_INFLUENCE))
        mf_location_to_tuo_neighbor_location_and_locations_influenced_score = \
            dict(Experiments.load_tuo_location_and_tuo_neighbor_location_and_locations_influence_score(model_id, hashtag_tag, noOfInfluencers, InfluenceMeasuringModels.TYPE_OUTGOING_INFLUENCE))
        for location in mf_location_to_tuo_neighbor_location_and_locations_influenced_score.keys()[:]:
            if not isWithinBoundingBox(getLocationFromLid(location.replace('_', ' ')), boundary):
                if location in mf_location_to_tuo_neighbor_location_and_locations_influencing_score:
                    del mf_location_to_tuo_neighbor_location_and_locations_influencing_score[location]
                del mf_location_to_tuo_neighbor_location_and_locations_influenced_score[location]
        no_of_locations = len(mf_location_to_tuo_neighbor_location_and_locations_influenced_score)
        for location, tuo_neighbor_location_and_locations_influencing_score in \
                mf_location_to_tuo_neighbor_location_and_locations_influencing_score.iteritems():
            mf_location_to_mf_influence_type_to_influence_score[location][InfluenceMeasuringModels.TYPE_INCOMING_INFLUENCE] \
                = sum(zip(*tuo_neighbor_location_and_locations_influencing_score)[1])/no_of_locations
        for location, tuo_neighbor_location_and_locations_influenced_score in \
                mf_location_to_tuo_neighbor_location_and_locations_influenced_score.iteritems():
            mf_location_to_mf_influence_type_to_influence_score[location][InfluenceMeasuringModels.TYPE_OUTGOING_INFLUENCE] \
                = sum(zip(*tuo_neighbor_location_and_locations_influenced_score)[1])/no_of_locations
        for location, mf_influence_type_to_influence_score in \
                mf_location_to_mf_influence_type_to_influence_score.iteritems():
            influence_type, influence_score = max(mf_influence_type_to_influence_score.iteritems(), key=itemgetter(1))
            if influence_type==InfluenceMeasuringModels.TYPE_INCOMING_INFLUENCE: 
                mf_location_to_global_influence_score[location] = -influence_score
            else: mf_location_to_global_influence_score[location] = influence_score
        return mf_location_to_global_influence_score.items()
    @staticmethod
    def load_tuo_location_and_mf_influence_type_to_influence_vector(model_id, vector_length=25):
        def convert_to_vector(tuo_location_and_influence_score):
            if vector_length: tuo_location_and_influence_score = sorted(
                                                               tuo_location_and_influence_score,
                                                               key = lambda (_, transmission_score): abs(transmission_score),
                                                               reverse=True
                                                        )[:vector_length]
            else: tuo_location_and_influence_score = sorted(
                                                               tuo_location_and_influence_score,
                                                               key = lambda (_, transmission_score): abs(transmission_score),
                                                               reverse=True
                                                        )                                      
            root_of_sum_of_squares = np.sqrt(sum([influence_score**2 for _, influence_score in tuo_location_and_influence_score]))
            return dict([(location, influence_score/root_of_sum_of_squares) for location, influence_score in tuo_location_and_influence_score])
        tuo_location_and_mf_influence_type_to_influence_vector = []
        tuo_location_and_tuo_neighbor_location_and_influence_score = Experiments.load_tuo_location_and_tuo_neighbor_location_and_influence_score(model_id)
        for location, tuo_neighbor_location_and_influence_score  in \
                tuo_location_and_tuo_neighbor_location_and_influence_score:
            tuo_outgoing_location_and_influence_score = filter(
                                                                lambda (neighbor_location, influence_score): influence_score>0,
                                                                tuo_neighbor_location_and_influence_score
                                                        )
            tuo_incoming_location_and_influence_score = filter(
                                                            lambda (neighbor_location, influence_score): influence_score<0, 
                                                            tuo_neighbor_location_and_influence_score
                                                        )
            location_influence_vector = convert_to_vector(tuo_neighbor_location_and_influence_score)
            location_outgoing_locations_vector = convert_to_vector(tuo_outgoing_location_and_influence_score)
            location_incoming_locations_vector = convert_to_vector(tuo_incoming_location_and_influence_score)
            tuo_location_and_mf_influence_type_to_influence_vector.append([location, {
                                                                            InfluenceMeasuringModels.TYPE_COMPLETE_INFLUENCE: location_influence_vector,
                                                                            InfluenceMeasuringModels.TYPE_OUTGOING_INFLUENCE: location_outgoing_locations_vector,
                                                                            InfluenceMeasuringModels.TYPE_INCOMING_INFLUENCE: location_incoming_locations_vector,
                                                                            }
                                                                ])
        return tuo_location_and_mf_influence_type_to_influence_vector
    @staticmethod
    def generate_tuo_location_and_tuo_neighbor_location_and_mf_influence_type_and_similarity(model_ids, startTime, endTime, outputFolder):
        def location_similarity(location_vector_1, location_vector_2): 
            return reduce(lambda total, k: total+(location_vector_1.get(k,0)*location_vector_2.get(k,0)), set(location_vector_1.keys()).union(location_vector_2.keys()),0.)
        influence_types=[InfluenceMeasuringModels.TYPE_COMPLETE_INFLUENCE, InfluenceMeasuringModels.TYPE_OUTGOING_INFLUENCE, InfluenceMeasuringModels.TYPE_INCOMING_INFLUENCE]
        for model_id in model_ids:
            mf_location_to_mf_influence_type_to_influence_vector = dict(Experiments.load_tuo_location_and_mf_influence_type_to_influence_vector(model_id))
            GeneralMethods.runCommand('rm -rf %s'%tuo_location_and_tuo_neighbor_location_and_mf_influence_type_and_similarity_file%model_id)
            for line_count, location_object in enumerate(iterateJsonFromFile(
                         location_objects_file%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
                     )):
                print line_count
                location = location_object['id']
                tuo_neighbor_location_and_mf_influence_type_and_similarity = []
                for neighbor_location in location_object['links'].keys(): 
                    mf_influence_type_and_similarity = {}
                    for influence_type in influence_types:
                        similarity = location_similarity( 
                                                             mf_location_to_mf_influence_type_to_influence_vector[location][influence_type],
                                                             mf_location_to_mf_influence_type_to_influence_vector[neighbor_location][influence_type]
                                                      )
                        mf_influence_type_and_similarity[influence_type] = similarity
                    so_hashtags_for_location = set(location_object['hashtags'].keys())
                    so_hashtags_for_neighbor_location = set(location_object['links'][neighbor_location].keys())
                    numerator = len(so_hashtags_for_location.intersection(so_hashtags_for_neighbor_location)) + 0.
                    denominator = len(so_hashtags_for_location.union(so_hashtags_for_neighbor_location)) + 0.
                    mf_influence_type_and_similarity[JACCARD_SIMILARITY] = numerator/denominator                
                    tuo_neighbor_location_and_mf_influence_type_and_similarity.append([neighbor_location, mf_influence_type_and_similarity])
                FileIO.writeToFileAsJson(
                                         [location, tuo_neighbor_location_and_mf_influence_type_and_similarity],
                                         tuo_location_and_tuo_neighbor_location_and_mf_influence_type_and_similarity_file%model_id
                                         )
    @staticmethod
    def generate_tuo_location_and_tuo_neighbor_location_and_sharing_affinity_score(model_ids, startTime, endTime, outputFolder):
        for model_id in model_ids:
            for line_count, location_object in enumerate(iterateJsonFromFile(
                         location_objects_file%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
                         )):
                print line_count
                mf_from_neighbor_location_to_sharing_affinity_score = {}
                so_hashtags = set(location_object['hashtags'])
                for neighbor_location, neighbor_hashtags in location_object['links'].iteritems():
                    so_neighbor_hashtags = set(neighbor_hashtags)
                    mf_from_neighbor_location_to_sharing_affinity_score[neighbor_location]=len(so_hashtags.intersection(so_neighbor_hashtags))/float(len(so_hashtags))
                FileIO.writeToFileAsJson([
                                          location_object['id'], 
                                          sorted(mf_from_neighbor_location_to_sharing_affinity_score.iteritems(), key=itemgetter(1), reverse=True)
                                        ], 
                                        tuo_location_and_tuo_neighbor_location_and_sharing_affinity_score_file%model_id)
    @staticmethod
    def load_tuo_location_and_tuo_neighbor_location_and_sharing_affinity_score(model_id):
        return [(location, tuo_neighbor_location_and_sharing_affinity_score)
                 for location, tuo_neighbor_location_and_sharing_affinity_score in 
                 iterateJsonFromFile(tuo_location_and_tuo_neighbor_location_and_sharing_affinity_score_file%model_id)]
    @staticmethod
    def load_ltuo_location_and_no_of_occurrences(START_TIME=datetime(2011, 5, 1), END_TIME=datetime(2011, 12, 31), WINDOW_OUTPUT_FOLDER='complete_prop'):
        ltuo_location_and_no_of_occurrences = []
        for location, ltuo_hashtag_and_occurrence_time in \
                iterateJsonFromFile(f_ltuo_location_and_ltuo_hashtag_and_occurrence_time%(WINDOW_OUTPUT_FOLDER, START_TIME.strftime('%Y-%m-%d'), END_TIME.strftime('%Y-%m-%d'))):
            ltuo_location_and_no_of_occurrences.append([location, len(ltuo_hashtag_and_occurrence_time)])
        return ltuo_location_and_no_of_occurrences
    @staticmethod
    def get_locations_sorted_by_boundary_influence_score(model_id, hashtag_tag, no_of_locations=None):
            ltuo_location_and_global_influence_score = Experiments.load_tuo_location_and_boundary_influence_score(model_id, hashtag_tag)
            if no_of_locations: return zip(*sorted(ltuo_location_and_global_influence_score, key=itemgetter(1)))[0][:no_of_locations]
            else: return zip(*sorted(ltuo_location_and_global_influence_score, key=itemgetter(1)))[0]
    @staticmethod
    def load_ltuo_hashtag_and_ltuo_location_and_occurrence_time(startTime=datetime(2012, 1, 1), endTime=datetime(2012, 3, 31), outputFolder='complete_prop'):
        ltuo_hashtag_and_ltuo_location_and_occurrence_time = []
        for hashtag_object in \
                iterateJsonFromFile(f_hashtag_objects%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))):
            ltuo_location_and_occurrence_time = [
                                                 (getLatticeLid(point, LOCATION_ACCURACY), occurrence_time)
                                                 for point, occurrence_time in hashtag_object['oc']
                                                 ]
            ltuo_hashtag_and_ltuo_location_and_occurrence_time.append([hashtag_object['h'], ltuo_location_and_occurrence_time])
        return ltuo_hashtag_and_ltuo_location_and_occurrence_time
    @staticmethod
    def generate_hashtag_specific_location_and_pure_influence_scores(test_models_ids):
        for test_model_id in test_models_ids:
            output_file = f_ltuo_hashtag_and_ltuo_location_and_pure_influence_score%(test_model_id)
            GeneralMethods.runCommand('rm -rf %s'%output_file)
            ltuo_hashtag_and_ltuo_location_and_occurrence_time = Experiments.load_ltuo_hashtag_and_ltuo_location_and_occurrence_time()
            for hashtag_count, (hashtag, ltuo_location_and_occurrence_time) in\
                    enumerate(ltuo_hashtag_and_ltuo_location_and_occurrence_time):
                ltuo_location_and_occurrence_times = [(location, sorted(zip(*ito_location_and_occurrence_time)[1]))
                                                        for location, ito_location_and_occurrence_time in
                                                            groupby(
                                                                    sorted(ltuo_location_and_occurrence_time, key=itemgetter(0)),
                                                                    key=itemgetter(0)
                                                            )
                                                    ] 
                print hashtag_count, test_model_id
                ltuo_location_and_pure_influence_score = []
                for location, location_occurrence_times in ltuo_location_and_occurrence_times:
                    pure_influence_scores = []
                    for neighbor_location, neighbor_location_occurrence_times in ltuo_location_and_occurrence_times:
                        if location!=neighbor_location:
                            pure_influence_score = MF_INFLUENCE_MEASURING_MODELS_TO_MODEL_ID[test_model_id](neighbor_location_occurrence_times, location_occurrence_times)
                            pure_influence_scores.append(pure_influence_score)
                    ltuo_location_and_pure_influence_score.append([location, np.mean(pure_influence_scores)])
                ltuo_location_and_pure_influence_score = sorted(ltuo_location_and_pure_influence_score, key=itemgetter(1))
                FileIO.writeToFileAsJson([hashtag, ltuo_location_and_pure_influence_score], output_file)
#                def _to_locations_based_on_first_occurence(locations, location):
#                    if location not in locations: locations.append(location)
#                    return locations
#                print reduce(_to_locations_based_on_first_occurence, zip(*sorted(ltuo_location_and_occurrence_time, key=itemgetter(1)))[0], [])[:5]
#                if hashtag_count==5: exit()
    @staticmethod
    def load_ltuo_test_hashtag_and_ltuo_location_and_pure_influence_score(test_model_id):
        return [ (hashtag, ltuo_location_and_pure_influence_score)
                 for hashtag, ltuo_location_and_pure_influence_score in 
                    iterateJsonFromFile(f_ltuo_hashtag_and_ltuo_location_and_pure_influence_score%(test_model_id))
                ]
    @staticmethod
    def _cluster_locations_based_on_influence_scores(ltuo_locations_and_influence_score):
        def similarity_matrix(similarity_matrix, (current_point, all_points)):
            similarity_matrix.append([1./(np.abs(current_point - point)+1)for point in all_points])
            return similarity_matrix
        locations, influence_scores = zip(*ltuo_locations_and_influence_score)
        S = np.array(reduce(
                        similarity_matrix,
                        zip(influence_scores, [influence_scores]*len(influence_scores)),
                        []
                    ))
        af = AffinityPropagation().fit(S)
        return (len(af.cluster_centers_indices_), zip(locations, influence_scores, af.labels_))
    @staticmethod
    def get_location_with_zone_ids(model_id, hashtag_tag):
        ltuo_location_and_global_influence_score = Experiments.load_tuo_location_and_boundary_influence_score(model_id, hashtag_tag)
        no_of_zones, ltuo_location_and_influence_score_and_zone_id \
            = Experiments._cluster_locations_based_on_influence_scores(ltuo_location_and_global_influence_score)
        # Combine singleton clusters
        locations, influence_scores, zone_ids = zip(*ltuo_location_and_influence_score_and_zone_id)
        ltuo_influence_score_and_zone_id = zip(influence_scores, zone_ids)
        ltuo_zone_id_and_no_of_locations = [(zone_id, len(list(ito_tuo_influence_score_and_zone_id)))
                                                for zone_id, ito_tuo_influence_score_and_zone_id in
                                                    groupby(
                                                            sorted(ltuo_influence_score_and_zone_id, key=itemgetter(1)),
                                                            key=itemgetter(1)
                                                    )
                                            ]
        singleton_zone_ids = [zone_id for zone_id, no_of_locations in ltuo_zone_id_and_no_of_locations if no_of_locations==1]
        temp_ltuo_location_and_influence_score_and_zone_id = []
        for location, influence_score, zone_id in \
                ltuo_location_and_influence_score_and_zone_id:
            if zone_id not in singleton_zone_ids: temp_ltuo_location_and_influence_score_and_zone_id.append((location, influence_score, zone_id))
            else: temp_ltuo_location_and_influence_score_and_zone_id.append((location, influence_score, Experiments.SINGLETON_ZONE_ID))
        ltuo_location_and_influence_score_and_zone_id = temp_ltuo_location_and_influence_score_and_zone_id
        no_of_zones=no_of_zones-len(singleton_zone_ids)+1
        # Rename zone ids
        locations, influence_scores, zone_ids = zip(*ltuo_location_and_influence_score_and_zone_id)
        ltuo_influence_score_and_zone_id = zip(influence_scores, zone_ids)
        ltuo_zone_id_and_min_influence_score = [(zone_id, min(ito_tuo_influence_score_and_zone_id))
                                                    for zone_id, ito_tuo_influence_score_and_zone_id in
                                                        groupby(
                                                                sorted(ltuo_influence_score_and_zone_id, key=itemgetter(1)),
                                                                key=itemgetter(1)
                                                        )
                                                ]
        ltuo_zone_id_and_new_zone_id = zip(
                                            zip(*sorted(ltuo_zone_id_and_min_influence_score, key=itemgetter(1)))[0],
                                            range(len(ltuo_zone_id_and_min_influence_score))
                                        )
        mf_zone_id_to_new_zone_id = dict(ltuo_zone_id_and_new_zone_id)
        temp_ltuo_location_and_influence_score_and_zone_id = []
        for location, influence_score, zone_id in \
                ltuo_location_and_influence_score_and_zone_id:
            temp_ltuo_location_and_influence_score_and_zone_id.append([location, influence_score, mf_zone_id_to_new_zone_id[zone_id]])
        ltuo_location_and_influence_score_and_zone_id = temp_ltuo_location_and_influence_score_and_zone_id
        
        return no_of_zones, ltuo_location_and_influence_score_and_zone_id
    @staticmethod
    def run():
        model_ids = [
#                      InfluenceMeasuringModels.ID_FIRST_OCCURRENCE, 
#                      InfluenceMeasuringModels.ID_MEAN_OCCURRENCE, 
#                      InfluenceMeasuringModels.ID_AGGREGATE_OCCURRENCE, 
                      InfluenceMeasuringModels.ID_WEIGHTED_AGGREGATE_OCCURRENCE,
                  ]
        hashtag_tag = wout_extra_hashtags_tag
        
        test_model_ids = [
                      InfluenceMeasuringModels.ID_WEIGHTED_AGGREGATE_OCCURRENCE,
                  ]
        
#        START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER = datetime(2011, 5, 1), datetime(2011, 12, 31), 'complete_prop' # Complete propagation duration
        
#        Experiments.generate_tuo_location_and_tuo_neighbor_location_and_pure_influence_score(model_ids, START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER, hashtag_tag)
#        Experiments.generate_tuo_location_and_tuo_neighbor_location_and_influence_score(model_ids, START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER, hashtag_tag)
#        Experiments.generate_tuo_location_and_tuo_neighbor_location_and_mf_influence_type_and_similarity(model_ids, START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER)
#        Experiments.generate_tuo_location_and_tuo_neighbor_location_and_sharing_affinity_score(model_ids, START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER)

        Experiments.generate_hashtag_specific_location_and_pure_influence_scores(test_model_ids)

if __name__ == '__main__':
    Experiments.run()
        
    
    