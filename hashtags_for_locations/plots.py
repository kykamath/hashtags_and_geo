'''
Created on Feb 27, 2012

@author: kykamath

How fast does each location learn?
Variation of learning with varying beta both performance and speed?
For both learning models.

Run these two methods:
plot_model_learning_time_series
plot_model_learning_time_on_map

'''
from operator import itemgetter
from analysis import iterateJsonFromFile
from itertools import groupby
from library.geo import getLocationFromLid, plotPointsOnWorldMap, \
            getLatticeLid, plot_graph_clusters_on_world_map, isWithinBoundingBox
from collections import defaultdict
import matplotlib.pyplot as plt
from library.classes import GeneralMethods
from models import loadLocationsList,\
    PredictionModels, Propagations, PREDICTION_MODEL_METHODS
from mr_analysis import LOCATION_ACCURACY
import random, matplotlib, inspect, time
from library.file_io import FileIO
from models import ModelSelectionHistory
from settings import analysisFolder, timeUnitWithOccurrencesFile, \
        PARTIAL_WORLD_BOUNDARY, hashtagsWithoutEndingWindowFile, \
        hashtagsWithoutEndingWindowWithoutLatticeApproximationFile
from datetime import datetime
from library.stats import getOutliersRangeUsingIRQ, filter_outliers
import numpy as np
from hashtags_for_locations.models import loadSharingProbabilities,\
    EvaluationMetrics
import networkx as nx
from library.graphs import clusterUsingAffinityPropagation
from scipy.stats import ks_2samp
from library.plotting import CurveFit, splineSmooth
import scipy
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
from mr_analysis import TIME_UNIT_IN_SECONDS

ALL_LOCATIONS = 'all_locations'
MAP_FROM_MODEL_TO_COLOR = dict([
                                (PredictionModels.COVERAGE_DISTANCE, 'b'), (PredictionModels.COVERAGE_PROBABILITY, 'm'), (PredictionModels.SHARING_PROBABILITY, 'r'), (PredictionModels.TRANSMITTING_PROBABILITY, 'k'),
                                (ModelSelectionHistory.FOLLOW_THE_LEADER, '#FF0A0A'), (ModelSelectionHistory.HEDGING_METHOD, '#9661FF'),
                                (PredictionModels.COMMUNITY_AFFINITY, '#436DFC'), (PredictionModels.SPATIAL, '#F15CFF'), (ALL_LOCATIONS, '#FFB44A')
                                ])
MAP_FROM_MODEL_TO_MARKER = dict([ 
                                 (ModelSelectionHistory.FOLLOW_THE_LEADER, 'd'), (ModelSelectionHistory.HEDGING_METHOD, 'o'),
                                 (PredictionModels.COMMUNITY_AFFINITY, 'x'), (PredictionModels.SPATIAL, 'o'), (ALL_LOCATIONS, 'd'),
                                 ])
MAP_FROM_MODEL_TO_MODEL_TYPE = dict([
                                     (PredictionModels.SHARING_PROBABILITY, PredictionModels.COMMUNITY_AFFINITY),
                                     (PredictionModels.TRANSMITTING_PROBABILITY, PredictionModels.COMMUNITY_AFFINITY),
                                     (PredictionModels.COVERAGE_DISTANCE, PredictionModels.SPATIAL),
                                     (PredictionModels.COVERAGE_PROBABILITY, PredictionModels.SPATIAL),
                                     ])
MAP_FROM_MODEL_TO_SUBPLOT_ID = dict([ (ModelSelectionHistory.FOLLOW_THE_LEADER, 211), (ModelSelectionHistory.HEDGING_METHOD, 212)])

def getHashtagColors(hashtag_and_occurrence_locations):
    return dict([(hashtag, GeneralMethods.getRandomColor()) for hashtag, occurrence_locations in hashtag_and_occurrence_locations if len(occurrence_locations)>0])

def plotAllData(prediction_models):
    loadLocationsList()
    for data in iterateJsonFromFile('mr_Data/timeUnitWithOccurrences'):
        hashtag_and_occurrence_locations = [(h, map(lambda lid: getLocationFromLid(lid.replace('_', ' ')), zip(*occs)[1])) for h, occs in groupby(sorted(data['oc'], key=itemgetter(0)), key=itemgetter(0))]
        map_from_hashtag_to_color = getHashtagColors(hashtag_and_occurrence_locations)
        tuple_of_hashtag_and__list_of__tuple_of_location_and_no_of_occurrences_of_hashtag = [(hashtag, [(location, len(list(occurrence_locations))) for location, occurrence_locations in groupby(occurrence_locations, key=itemgetter(0,1))]) for hashtag, occurrence_locations in hashtag_and_occurrence_locations]
        map_from_locations_to__list_of__tuple_of_hashtag_and_no_of_occurrences_of_hashtag = defaultdict(list)
        for hashtag, list_of__tuple_of_location_and_no_of_occurrences_of_hashtag in tuple_of_hashtag_and__list_of__tuple_of_location_and_no_of_occurrences_of_hashtag:
            for location, no_of_occurrences_of_hashtag in list_of__tuple_of_location_and_no_of_occurrences_of_hashtag:
                map_from_locations_to__list_of__tuple_of_hashtag_and_no_of_occurrences_of_hashtag[location].append((hashtag, no_of_occurrences_of_hashtag))
        if len(map_from_locations_to__list_of__tuple_of_hashtag_and_no_of_occurrences_of_hashtag)>100:
            conf = {'noOfTargetHashtags': 1}
            propagation = Propagations(None, None)
            list_of_tuple_of_hashtag_and_location_and_time = [(hashtag, getLatticeLid(location, accuracy=LOCATION_ACCURACY), None) for hashtag, occurrence_locations in hashtag_and_occurrence_locations for location in occurrence_locations]
            propagation.update(list_of_tuple_of_hashtag_and_location_and_time)
            #Plot real data.
            plt.figure()
            tuple_of_location_and__tuple_of_hashtag_and_no_of_occurrences = [(location, max(list_of__tuple_of_hashtag_and_no_of_occurrences_of_hashtag, key=itemgetter(1))) for location, list_of__tuple_of_hashtag_and_no_of_occurrences_of_hashtag in map_from_locations_to__list_of__tuple_of_hashtag_and_no_of_occurrences_of_hashtag.iteritems()]
            locations, colors = zip(*[(location, map_from_hashtag_to_color[hashtag]) for location, (hashtag, no_of_occurrences) in tuple_of_location_and__tuple_of_hashtag_and_no_of_occurrences if hashtag in map_from_hashtag_to_color])
            plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=colors, lw = 0)
            plt.title('observed_data')
            for prediction_model in prediction_models:
                plt.figure()
                map_from_location_to_list_of_predicted_hashtags = PREDICTION_MODEL_METHODS[prediction_model](propagation, **conf)
                list_of_tuple_of_location_and_color = [(getLocationFromLid(location.replace('_', ' ')), map_from_hashtag_to_color[hashtags[0]]) for location, hashtags in map_from_location_to_list_of_predicted_hashtags.iteritems() if hashtags and hashtags[0] in map_from_hashtag_to_color]
                locations, colors = zip(*list_of_tuple_of_location_and_color)
                plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=colors, lw = 0)
                plt.title(prediction_model)
            plt.show()
class GeneralAnalysis():
    @staticmethod
    def grid_visualization():
        BIN_ACCURACY = 1.45
#        BIN_ACCURACY = 0.145
        map_from_location_bin_to_color = {}
        set_of_observed_location_ids = set()
        tuples_of_location_and_bin_color = []
        for count, data in enumerate(iterateJsonFromFile(hashtagsWithoutEndingWindowWithoutLatticeApproximationFile%('testing', '2011-09-01', '2011-11-01'))):
            for location, time in data['oc']:
                location_id = getLatticeLid(location, LOCATION_ACCURACY)
                if location_id not in set_of_observed_location_ids:
                    set_of_observed_location_ids.add(location_id)
                    location_bin = getLatticeLid(location, BIN_ACCURACY)
                    if location_bin not in map_from_location_bin_to_color: map_from_location_bin_to_color[location_bin] = GeneralMethods.getRandomColor()
                    tuples_of_location_and_bin_color.append((location, map_from_location_bin_to_color[location_bin]))
            print count
            if count==1000: break
        locations, colors = zip(*tuples_of_location_and_bin_color)
        plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=colors, lw = 0)
#        plt.show()
        file_learning_analysis = './images/%s.png'%(GeneralMethods.get_method_id())
        FileIO.createDirectoryForFile(file_learning_analysis)
        plt.savefig(file_learning_analysis)
        
def follow_the_leader_method(map_from_model_to_weight): return min(map_from_model_to_weight.iteritems(), key=itemgetter(1))[0]
def hedging_method(map_from_model_to_weight):
    total_weight = sum(map_from_model_to_weight.values())
    for model in map_from_model_to_weight.keys(): map_from_model_to_weight[model]/=total_weight 
    tuple_of_id_model_and_cumulative_losses = [(id, model, cumulative_loss) for id, (model, cumulative_loss) in enumerate(map_from_model_to_weight.iteritems())]
    selected_id = GeneralMethods.weightedChoice(zip(*tuple_of_id_model_and_cumulative_losses)[2])
    return filter(lambda (id, model, _): id==selected_id, tuple_of_id_model_and_cumulative_losses)[0][1]
class LearningAnalysis():
    MAP_FROM_LEARNING_TYPE_TO_MODEL_SELECION_METHOD = dict([(ModelSelectionHistory.FOLLOW_THE_LEADER, follow_the_leader_method), (ModelSelectionHistory.HEDGING_METHOD, hedging_method)])
    @staticmethod
    def _get_flipping_ratio_for_all_locations(learning_type, no_of_hashtags):
        def count_non_flips((reduced_no_of_non_flips, reduced_previously_selected_model), (current_ep_time_unit, current_selected_model)):
            if reduced_previously_selected_model==current_selected_model: reduced_no_of_non_flips+=1.0 
            return (reduced_no_of_non_flips, current_selected_model)
        map_from_location_to_tuples_of_ep_time_unit_and_selected_model = defaultdict(list)
        tuples_of_location_and_flipping_ratio = []
        input_weight_file = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/testing/models/2011-09-01_2011-11-01/30_60/%s/%s_weights'%(no_of_hashtags, learning_type)
        set_of_ep_time_units = set()
        for data in iterateJsonFromFile(input_weight_file):
            if data['metricId']==EvaluationMetrics.ACCURACY:
                map_from_location_to_map_from_model_to_weight = data['location_weights']
                ep_time_unit = data['tu']
                set_of_ep_time_units.add(ep_time_unit)
                for location, map_from_model_to_weight in map_from_location_to_map_from_model_to_weight.iteritems():
                    selected_model = MAP_FROM_MODEL_TO_MODEL_TYPE[LearningAnalysis.MAP_FROM_LEARNING_TYPE_TO_MODEL_SELECION_METHOD[learning_type](map_from_model_to_weight)]
                    map_from_location_to_tuples_of_ep_time_unit_and_selected_model[location].append([ep_time_unit, selected_model])
        total_no_of_time_units = len(set_of_ep_time_units)
        for location, tuples_of_ep_time_unit_and_selected_model in map_from_location_to_tuples_of_ep_time_unit_and_selected_model.iteritems():
            if len(tuples_of_ep_time_unit_and_selected_model)>total_no_of_time_units:
                print 'x'
            non_flips_for_location, _ = reduce(count_non_flips, tuples_of_ep_time_unit_and_selected_model, (0.0, None))
            tuples_of_location_and_flipping_ratio.append([location, 1.0-(non_flips_for_location/total_no_of_time_units) ])
        return tuples_of_location_and_flipping_ratio
    @staticmethod
    def model_distribution_on_world_map(learning_type, no_of_hashtags, generate_data=True):
        weights_analysis_file = analysisFolder%'learning_analysis'+'/%s_weights_analysis'%(learning_type)
        if generate_data:
            GeneralMethods.runCommand('rm -rf %s'%weights_analysis_file)
            input_weight_file = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/testing/models/2011-09-01_2011-11-01/30_60/%s/%s_weights'%(no_of_hashtags, learning_type)
            final_map_from_location_to_map_from_model_to_weight = {}
            for data in iterateJsonFromFile(input_weight_file):
                map_from_location_to_map_from_model_to_weight = data['location_weights']
                for location, map_from_model_to_weight in map_from_location_to_map_from_model_to_weight.iteritems():
                    final_map_from_location_to_map_from_model_to_weight[location] = map_from_model_to_weight
            tuples_of_location_and_best_model = []
            for location, map_from_model_to_weight in final_map_from_location_to_map_from_model_to_weight.iteritems():
                tuples_of_weight_and_list_of_model_with_this_weight = [(weight, zip(*iterator_for_tuples_of_model_and_weight)[0]) 
                                                                        for weight, iterator_for_tuples_of_model_and_weight in 
                                                                        groupby(
                                                                                sorted(map_from_model_to_weight.iteritems(), key=itemgetter(1)), 
                                                                                key=itemgetter(1)
                                                                                )
                                                                       ]
                list_of_models_with_this_weight = min(tuples_of_weight_and_list_of_model_with_this_weight, key=itemgetter(0))[1]
                list_of_models_with_this_weight = set(map(lambda model: MAP_FROM_MODEL_TO_MODEL_TYPE[model], list_of_models_with_this_weight))
                if len(list_of_models_with_this_weight)==1: 
                    tuples_of_location_and_best_model.append((location, random.sample(list_of_models_with_this_weight,1)[0]))
            for tuple_of_location_and_best_model in tuples_of_location_and_best_model: FileIO.writeToFileAsJson(tuple_of_location_and_best_model, weights_analysis_file)
            print [(model, len(list(iterator_for_models))) for model, iterator_for_models in groupby(sorted(zip(*tuples_of_location_and_best_model)[1]))]
        else:
            tuples_of_location_and_best_model = [tuple_of_location_and_best_model for tuple_of_location_and_best_model in FileIO.iterateJsonFromFile(weights_analysis_file)]
            tuples_of_model_and_locations = [(model, zip(*iterator_of_tuples_of_location_and_models)[0]) 
                                           for model, iterator_of_tuples_of_location_and_models in 
                                           groupby(
                                                  sorted(tuples_of_location_and_best_model, key=itemgetter(1)),
                                                  key=itemgetter(1)
                                                  )
                                           ]
            for model, locations in tuples_of_model_and_locations:
                locations = [getLocationFromLid(location.replace('_', ' ')) for location in locations if isWithinBoundingBox(getLocationFromLid(location.replace('_', ' ')), PARTIAL_WORLD_BOUNDARY)]
                plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=MAP_FROM_MODEL_TO_COLOR[model], lw = 0)
    #            plt.show()
#                plt.savefig('images/learning_analysis/%s.png'%model)
#                plt.clf()
                file_learning_analysis = './images/%s_%s.png'%(GeneralMethods.get_method_id(), model)
                FileIO.createDirectoryForFile(file_learning_analysis)
                plt.savefig(file_learning_analysis)
#                plt.show()
                plt.clf()
    @staticmethod
    def correlation_between_model_type_and_location_size(learning_type):
        NO_OF_OCCURRENCES_BIN_SIZE = 2000
        weights_analysis_file = analysisFolder%'learning_analysis'+'/%s_weights_analysis'%(learning_type)
        tuples_of_location_and_best_model = [tuple_of_location_and_best_model for tuple_of_location_and_best_model in FileIO.iterateJsonFromFile(weights_analysis_file)]
        map_from_location_to_best_model = dict(tuples_of_location_and_best_model)
        
#        startTime, endTime, outputFolder = datetime(2011, 9, 1), datetime(2011, 11, 1), 'testing'
        startTime, endTime, outputFolder = datetime(2011, 4, 1), datetime(2012, 1, 31), 'complete' # Complete duration
        input_file = timeUnitWithOccurrencesFile%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
        map_from_location_to_no_of_occurrences_at_location = defaultdict(float)
        for time_unit_object in iterateJsonFromFile(input_file):
            for (_, location, _) in time_unit_object['oc']: 
                if location in map_from_location_to_best_model: map_from_location_to_no_of_occurrences_at_location[location]+=1.0
                
        map_from_model_to_tuples_of_location_and_no_of_occurrences_at_location = dict([(model, [(location, map_from_location_to_no_of_occurrences_at_location[location]) for location in zip(*iterator_of_tuples_of_location_and_models)[0]]) 
                                                                                       for model, iterator_of_tuples_of_location_and_models in 
                                                                                       groupby(
                                                                                              sorted(tuples_of_location_and_best_model, key=itemgetter(1)),
                                                                                              key=itemgetter(1)
                                                                                              )
                                                                                   ])
        map_from_model_to_tuples_of_location_and_no_of_occurrences_at_location[ALL_LOCATIONS] = map_from_location_to_no_of_occurrences_at_location.items()
    #    for model, tuples_of_location_and_no_of_occurrences_at_location in map_from_model_to_tuples_of_location_and_no_of_occurrences_at_location.items()[:]: print model, len(tuples_of_location_and_no_of_occurrences_at_location)
#        for model, tuples_of_location_and_no_of_occurrences_at_location in map_from_model_to_tuples_of_location_and_no_of_occurrences_at_location.items()[:]:
#            _, upper_range_for_no_of_occurrences_at_location = getOutliersRangeUsingIRQ(zip(*tuples_of_location_and_no_of_occurrences_at_location)[1])
#            map_from_model_to_tuples_of_location_and_no_of_occurrences_at_location[model] = filter(
#                                                                                                   lambda (location, no_of_occurrences_at_location): no_of_occurrences_at_location < upper_range_for_no_of_occurrences_at_location, 
#                                                                                                   tuples_of_location_and_no_of_occurrences_at_location)
        tuples_of_model_and_tuples_of_location_and_no_of_occurrences_at_location = map_from_model_to_tuples_of_location_and_no_of_occurrences_at_location.items()
#        for model, tuples_of_location_and_no_of_occurrences_at_location in tuples_of_model_and_tuples_of_location_and_no_of_occurrences_at_location:
#            print model, ks_2samp(zip(*map_from_model_to_tuples_of_location_and_no_of_occurrences_at_location[ALL_LOCATIONS])[1], list(zip(*tuples_of_location_and_no_of_occurrences_at_location)[1]))
            
        for model, tuples_of_location_and_no_of_occurrences_at_location in tuples_of_model_and_tuples_of_location_and_no_of_occurrences_at_location:
            print model, len(tuples_of_location_and_no_of_occurrences_at_location)
            total_no_of_locations = float(len(tuples_of_location_and_no_of_occurrences_at_location))
            list_of_no_of_occurrences_at_location = zip(*tuples_of_location_and_no_of_occurrences_at_location)[1]
#            list_of_no_of_occurrences_at_location = filter_outliers(list_of_no_of_occurrences_at_location)
            tuples_of_bin_of_no_of_occurrences_at_location_and_no_of_occurrences_at_location = [((int(no_of_occurrences_at_location/NO_OF_OCCURRENCES_BIN_SIZE)*NO_OF_OCCURRENCES_BIN_SIZE)+ NO_OF_OCCURRENCES_BIN_SIZE, 
                                                                                               no_of_occurrences_at_location) 
                                                                                               for no_of_occurrences_at_location in list_of_no_of_occurrences_at_location]
            tuples_of_bin_of_no_of_occurrences_at_location_and_distribution = [(bin_of_no_of_occurrences_at_location, len(list(iterator_for_tuples_of_bin_of_no_of_occurrences_at_location_and_no_of_occurrences_at_location)))
                                                                                for bin_of_no_of_occurrences_at_location, iterator_for_tuples_of_bin_of_no_of_occurrences_at_location_and_no_of_occurrences_at_location in
                                                                                    groupby(
                                                                                        sorted(
                                                                                               tuples_of_bin_of_no_of_occurrences_at_location_and_no_of_occurrences_at_location,
                                                                                               key=itemgetter(0)
                                                                                         ),
                                                                                        key=itemgetter(0)
                                                                                    )
                                                                               ]
            
            x_bin_of_no_of_occurrences, y_distribution = zip(*[(bin_of_no_of_occurrences_at_location, distribution/total_no_of_locations)
                                                               for bin_of_no_of_occurrences_at_location, distribution in
                                                               sorted(
                                                                      tuples_of_bin_of_no_of_occurrences_at_location_and_distribution,
                                                                      key=itemgetter(0)
                                                                      )
                                                               ])
#            print zip(x_bin_of_no_of_occurrences, y_distribution)
#            x_bin_of_no_of_occurrences, y_distribution = splineSmooth(x_bin_of_no_of_occurrences, y_distribution)
            plt.semilogx(x_bin_of_no_of_occurrences, y_distribution, c=MAP_FROM_MODEL_TO_COLOR[model], label=model, marker=MAP_FROM_MODEL_TO_MARKER[model], lw=2)
        plt.legend()
        plt.xlabel('No. of occurrences', fontsize=20), plt.ylabel('Percentage of locations', fontsize=20)
#        plt.show()
        file_learning_analysis = './images/%s.png'%GeneralMethods.get_method_id()
        FileIO.createDirectoryForFile(file_learning_analysis)
        plt.savefig(file_learning_analysis)
        plt.clf()
    @staticmethod
    def model_learning_graphs_on_world_map(learning_type):
        def plot_graph_clusters_on_world_map1(graph_of_locations, s=0, lw=0, alpha=0.6, bkcolor='#CFCFCF', *args, **kwargs):  
            no_of_clusters, tuples_of_location_and_cluster_id = clusterUsingAffinityPropagation(graph_of_locations)
            map_from_location_to_cluster_id = dict(tuples_of_location_and_cluster_id)
            map_from_cluster_id_to_cluster_color = dict([(i, GeneralMethods.getRandomColor()) for i in range(no_of_clusters)])
            tuples_of_cluster_id_and_locations_in_cluster = [(cluster_id, zip(*iterator_of_tuples_of_location_and_cluster_id)[0]) 
                                                            for cluster_id, iterator_of_tuples_of_location_and_cluster_id in 
                                                                groupby(
                                                                        sorted(tuples_of_location_and_cluster_id, key=itemgetter(1)), 
                                                                        key=itemgetter(1)
                                                                        )
                                                         ]
            nodes_in_order, edges_in_order = [], []
            for cluster_id, locations_in_cluster in sorted(tuples_of_cluster_id_and_locations_in_cluster, key=lambda (_, locations_in_cluster): len(locations_in_cluster)):
                subgraph_of_locations = nx.subgraph(graph_of_locations, locations_in_cluster)
                nodes_in_order+=subgraph_of_locations.nodes()
                edges_in_order+=subgraph_of_locations.edges(data=True)
                
            points, colors = zip(*map(lambda  location: (getLocationFromLid(location.replace('_', ' ')), map_from_cluster_id_to_cluster_color[map_from_location_to_cluster_id[location]]), nodes_in_order))
            _, m = plotPointsOnWorldMap(points, c=colors, s=s, lw=lw, returnBaseMapObject=True,  *args, **kwargs)
            for u, v, data in edges_in_order:
                if map_from_location_to_cluster_id[u]==map_from_location_to_cluster_id[v]:
                    color, u, v, w = map_from_cluster_id_to_cluster_color[map_from_location_to_cluster_id[u]], getLocationFromLid(u.replace('_', ' ')), getLocationFromLid(v.replace('_', ' ')), data['w']
                    m.drawgreatcircle(u[1], u[0], v[1], v[0], color=color, alpha=alpha)
            return (no_of_clusters, tuples_of_location_and_cluster_id)
        
        map_from_location_to_map_from_neighboring_location_to_similarity_between_locations = loadSharingProbabilities()['neighborProbability']
        weights_analysis_file = analysisFolder%'learning_analysis'+'/%s_weights_analysis'%(learning_type)
        tuples_of_location_and_best_model = [tuple_of_location_and_best_model for tuple_of_location_and_best_model in FileIO.iterateJsonFromFile(weights_analysis_file)]
        tuples_of_model_and_locations = [(model, zip(*iterator_of_tuples_of_location_and_models)[0]) 
                                                                                       for model, iterator_of_tuples_of_location_and_models in 
                                                                                       groupby(
                                                                                              sorted(tuples_of_location_and_best_model, key=itemgetter(1)),
                                                                                              key=itemgetter(1)
                                                                                              )
                                                                                   ]
        for model, locations in tuples_of_model_and_locations:
            graph_of_locations = nx.Graph()
            for location in locations:
                for neighboring_location, similarity_between_locations in map_from_location_to_map_from_neighboring_location_to_similarity_between_locations[location].iteritems():
                    if location!=neighboring_location \
                        and isWithinBoundingBox(getLocationFromLid(location.replace('_', ' ')), PARTIAL_WORLD_BOUNDARY) \
                        and isWithinBoundingBox(getLocationFromLid(neighboring_location.replace('_', ' ')), PARTIAL_WORLD_BOUNDARY):
                            if not graph_of_locations.has_edge(location, neighboring_location): graph_of_locations.add_edge(location, neighboring_location, {'w': similarity_between_locations})
                            else: graph_of_locations[location][neighboring_location]['w']=min([similarity_between_locations, graph_of_locations[location][neighboring_location]['w']])
    
            no_of_clusters, _ = plot_graph_clusters_on_world_map1(graph_of_locations)
            file_learning_analysis = './images/%s_%s.png'%(GeneralMethods.get_method_id(), model)
            FileIO.createDirectoryForFile(file_learning_analysis)
            plt.savefig(file_learning_analysis)
            plt.clf()
    @staticmethod
    def learner_flipping_time_series(learning_types, no_of_hashtags):
        for learning_type in learning_types:
            input_weight_file = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/testing/models/2011-09-01_2011-11-01/30_60/%s/%s_weights'%(no_of_hashtags, learning_type)
            map_from_ep_time_unit_to_no_of_locations_that_didnt_flip = {}
            map_from_location_to_previously_selected_model = {}
            for data in iterateJsonFromFile(input_weight_file):
                if data['metricId']==EvaluationMetrics.ACCURACY:
                    map_from_location_to_map_from_model_to_weight = data['location_weights']
                    ep_time_unit = data['tu']
                    no_of_locations_that_didnt_flip = 0.0 
                    for location, map_from_model_to_weight in map_from_location_to_map_from_model_to_weight.iteritems():
        #                model_selected = LearningAnalysis.MAP_FROM_LEARNING_TYPE_TO_MODEL_SELECION_METHOD[learning_type](map_from_model_to_weight)
                        model_selected = MAP_FROM_MODEL_TO_MODEL_TYPE[LearningAnalysis.MAP_FROM_LEARNING_TYPE_TO_MODEL_SELECION_METHOD[learning_type](map_from_model_to_weight)]
                        if location in map_from_location_to_previously_selected_model and map_from_location_to_previously_selected_model[location]==model_selected: 
                            no_of_locations_that_didnt_flip+=1
                        map_from_location_to_previously_selected_model[location] = model_selected
                    map_from_ep_time_unit_to_no_of_locations_that_didnt_flip[ep_time_unit] = no_of_locations_that_didnt_flip
            total_no_of_locations = len(map_from_location_to_previously_selected_model)
            tuples_of_ep_time_unit_and_percentage_of_locations_that_flipped = [(ep_time_unit, 1.0 - (map_from_ep_time_unit_to_no_of_locations_that_didnt_flip[ep_time_unit]/total_no_of_locations)) 
                                                                               for ep_time_unit in sorted(map_from_ep_time_unit_to_no_of_locations_that_didnt_flip)
                                                                            ]
            ep_first_time_unit = tuples_of_ep_time_unit_and_percentage_of_locations_that_flipped[0][0]
            x_data, y_data = zip(*tuples_of_ep_time_unit_and_percentage_of_locations_that_flipped)
            x_data, y_data = splineSmooth(x_data, y_data)
            plt.plot([(x-ep_first_time_unit)/(60*60) for x in x_data], y_data, c=MAP_FROM_MODEL_TO_COLOR[learning_type], label=learning_type, lw=2, marker = MAP_FROM_MODEL_TO_MARKER[learning_type])
        plt.legend()
        plt.xlabel('Learning lag (hours)', fontsize=20), plt.ylabel('Percentage of locations that flipped', fontsize=20)
#        plt.show()
        file_learning_analysis = './images/%s.png'%GeneralMethods.get_method_id()
        FileIO.createDirectoryForFile(file_learning_analysis)
        plt.savefig(file_learning_analysis)
        plt.clf()
    @staticmethod
    def flipping_ratio_on_world_map(learning_types, no_of_hashtags):
        for learning_type in learning_types:
            tuples_of_location_and_flipping_ratio = LearningAnalysis._get_flipping_ratio_for_all_locations(learning_type, no_of_hashtags)
            locations, colors = zip(*[(getLocationFromLid(location.replace('_', ' ')), color)
                                      for location, color in sorted(tuples_of_location_and_flipping_ratio, key=itemgetter(1))])
            plt.subplot(MAP_FROM_MODEL_TO_SUBPLOT_ID[learning_type])
            sc = plotPointsOnWorldMap(locations, c=colors, cmap=matplotlib.cm.cool, lw = 0, alpha=1.0)
            plt.title(learning_type)
            plt.colorbar(sc)
#        plt.show()
        file_learning_analysis = './images/%s.png'%GeneralMethods.get_method_id()
        FileIO.createDirectoryForFile(file_learning_analysis)
        plt.savefig(file_learning_analysis)
        plt.clf()
    @staticmethod
    def flipping_ratio_correlation_with_no_of_occurrences_at_location(learning_types, no_of_hashtags):
        for learning_type in learning_types:
            NO_OF_OCCURRENCES_BIN_SIZE= 2000
            # Load flipping ratio data.
            map_from_location_to_flipping_ratio = dict(LearningAnalysis._get_flipping_ratio_for_all_locations(learning_type, no_of_hashtags))
            # Load no. of occurrences data
            #        startTime, endTime, outputFolder = datetime(2011, 9, 1), datetime(2011, 11, 1), 'testing'
            startTime, endTime, outputFolder = datetime(2011, 4, 1), datetime(2012, 1, 31), 'complete' # Complete duration
            input_file = timeUnitWithOccurrencesFile%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
            map_from_location_to_no_of_occurrences_at_location = defaultdict(float)
            for time_unit_object in iterateJsonFromFile(input_file):
                for (_, location, _) in time_unit_object['oc']: 
                    if location in map_from_location_to_flipping_ratio: map_from_location_to_no_of_occurrences_at_location[location]+=1
            tuples_of_location_and_flipping_ratio_and_no_of_occurrences_at_location = [(
                                                                                        location, 
                                                                                        map_from_location_to_flipping_ratio[location],
                                                                                        map_from_location_to_no_of_occurrences_at_location[location],
                                                                                        )
                                                                                       for location in map_from_location_to_no_of_occurrences_at_location]
            # Filter locations for no. of occurrences.
            no_of_occurrences_at_location = zip(*tuples_of_location_and_flipping_ratio_and_no_of_occurrences_at_location)[2]
            print len(tuples_of_location_and_flipping_ratio_and_no_of_occurrences_at_location)
            _, upper_range_no_of_occurrences_at_location = getOutliersRangeUsingIRQ(no_of_occurrences_at_location)
            tuples_of_location_and_flipping_ratio_and_no_of_occurrences_at_location = filter(lambda (_,__,no_of_occurrences_at_location): 
                                                                                                no_of_occurrences_at_location<=upper_range_no_of_occurrences_at_location,
                                                                                             tuples_of_location_and_flipping_ratio_and_no_of_occurrences_at_location)
            print len(tuples_of_location_and_flipping_ratio_and_no_of_occurrences_at_location)
            # Bin no. of occurrences.
            map_from_no_of_occurrences_at_location_bin_to_flipping_ratios = defaultdict(list)
            for _, flipping_ratio, no_of_occurrences_at_location in tuples_of_location_and_flipping_ratio_and_no_of_occurrences_at_location:
                no_of_occurrences_at_location_bin = int(no_of_occurrences_at_location/NO_OF_OCCURRENCES_BIN_SIZE)*NO_OF_OCCURRENCES_BIN_SIZE + NO_OF_OCCURRENCES_BIN_SIZE + 0.0
                map_from_no_of_occurrences_at_location_bin_to_flipping_ratios[no_of_occurrences_at_location_bin].append(flipping_ratio)
            for no_of_occurrences_at_location_bin in sorted(map_from_no_of_occurrences_at_location_bin_to_flipping_ratios):
                flipping_ratios = map_from_no_of_occurrences_at_location_bin_to_flipping_ratios[no_of_occurrences_at_location_bin]
                flipping_ratios = filter_outliers(flipping_ratios)
                if len(flipping_ratios) >= 5: map_from_no_of_occurrences_at_location_bin_to_flipping_ratios[no_of_occurrences_at_location_bin] = flipping_ratios
                else: del map_from_no_of_occurrences_at_location_bin_to_flipping_ratios[no_of_occurrences_at_location_bin]
            # Plot data.
            x_no_of_occurrences_at_location_bins, y_mean_flipping_ratios = zip(*[ (no_of_occurrences_at_location_bin, np.mean(flipping_ratios)) 
                  for no_of_occurrences_at_location_bin, flipping_ratios in 
                  sorted(map_from_no_of_occurrences_at_location_bin_to_flipping_ratios.iteritems(), key=itemgetter(0))
                  ])
            pearsonCoeff, p_value = scipy.stats.pearsonr(x_no_of_occurrences_at_location_bins, y_mean_flipping_ratios)
            print round(pearsonCoeff,2), round(p_value, 2)
            x_no_of_occurrences_at_location_bins, y_mean_flipping_ratios = np.array(list(x_no_of_occurrences_at_location_bins)), np.array(list(y_mean_flipping_ratios))
            parameters_after_fitting = CurveFit.getParamsAfterFittingData(x_no_of_occurrences_at_location_bins, y_mean_flipping_ratios, CurveFit.lineFunction, [0., 0.])
            y_fitted_mean_flipping_ratios = CurveFit.getYValues(CurveFit.lineFunction, parameters_after_fitting, x_no_of_occurrences_at_location_bins)
            plt.scatter(x_no_of_occurrences_at_location_bins, y_mean_flipping_ratios, lw=0, c=MAP_FROM_MODEL_TO_COLOR[learning_type], label=learning_type, marker=MAP_FROM_MODEL_TO_MARKER[learning_type])
            plt.plot(x_no_of_occurrences_at_location_bins, y_fitted_mean_flipping_ratios, lw=2, c=MAP_FROM_MODEL_TO_COLOR[learning_type])
        plt.xlabel('No. of occurrences', fontsize=20), plt.ylabel('Flipping ratio', fontsize=20)
#        plt.show()
        plt.legend()
        file_learning_analysis = './images/%s.png'%GeneralMethods.get_method_id()
        FileIO.createDirectoryForFile(file_learning_analysis)
        plt.savefig(file_learning_analysis)
        plt.clf()
            
        
    @staticmethod
    def run():
        no_of_hashtags = 4
        LearningAnalysis.model_distribution_on_world_map(learning_type=ModelSelectionHistory.FOLLOW_THE_LEADER, no_of_hashtags=no_of_hashtags, generate_data=False)
#        LearningAnalysis.correlation_between_model_type_and_location_size(learning_type=ModelSelectionHistory.FOLLOW_THE_LEADER)
#        LearningAnalysis.model_learning_graphs_on_world_map(learning_type=ModelSelectionHistory.FOLLOW_THE_LEADER)
#        LearningAnalysis.learner_flipping_time_series([ModelSelectionHistory.FOLLOW_THE_LEADER, ModelSelectionHistory.HEDGING_METHOD], no_of_hashtags)
#        LearningAnalysis.flipping_ratio_on_world_map([ModelSelectionHistory.FOLLOW_THE_LEADER, ModelSelectionHistory.HEDGING_METHOD], no_of_hashtags)
#        LearningAnalysis.flipping_ratio_correlation_with_no_of_occurrences_at_location([ModelSelectionHistory.FOLLOW_THE_LEADER, ModelSelectionHistory.HEDGING_METHOD], no_of_hashtags)
            
            
class PaperPlots:
    @staticmethod
    def hashtag_ditribution_on_world_map_by_time_units():
        currentTime, end_time = datetime(2011, 9, 1), datetime(2011, 11, 1)
        historyTimeInterval = timedelta(seconds=12*TIME_UNIT_IN_SECONDS)
        predictionTimeInterval = timedelta(seconds=2*TIME_UNIT_IN_SECONDS)
        output_file = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/hashtag_ditribution_on_world_map_by_time_units/%s/%s.png'
        timeUnitDelta = timedelta(seconds=TIME_UNIT_IN_SECONDS)
        historicalTimeUnitsMap, predictionTimeUnitsMap = {}, {}
        loadLocationsList()
        time_unit_with_occurrences_file = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/testing/2011-09-01_2011-11-01/timeUnitWithOccurrences'
        print 'Using file: ', time_unit_with_occurrences_file 
        timeUnitsToDataMap = dict([(d['tu'], d) for d in iterateJsonFromFile(time_unit_with_occurrences_file)])
        while currentTime<end_time:
            print currentTime, historyTimeInterval.seconds/60#, self.predictionTimeInterval.seconds/60
            currentOccurrences = []
            currentTimeObject = timeUnitsToDataMap.get(time.mktime(currentTime.timetuple()), {})
            if currentTimeObject: currentOccurrences=currentTimeObject['oc']
            for i in range(historyTimeInterval.seconds/TIME_UNIT_IN_SECONDS):
                historicalTimeUnit = currentTime-i*timeUnitDelta
                if historicalTimeUnit not in historicalTimeUnitsMap: historicalTimeUnitsMap[historicalTimeUnit]=Propagations(historicalTimeUnit, historyTimeInterval)
                historicalTimeUnitsMap[historicalTimeUnit].update(currentOccurrences)
            timeUnitForActualPropagation = currentTime-predictionTimeInterval
            timeUnitForPropagationForPrediction = timeUnitForActualPropagation-historyTimeInterval
            if timeUnitForPropagationForPrediction in historicalTimeUnitsMap:
                tuples_of_location_and_hashtag_and_occurrence_time = []
                for location, tuples_of_hashtag_and_occurrence_time in historicalTimeUnitsMap[timeUnitForPropagationForPrediction].occurrences.iteritems():
                    tuples_of_location_and_hashtag_and_occurrence_time+= [[getLocationFromLid(location.replace('_', ' ')), hashtag, occurrence_time]
                                                             for hashtag, occurrence_time in tuples_of_hashtag_and_occurrence_time]
                    
                tuples_of_hashtag_and_tuples_of_location_and_hashtag_and_occurrence_time = [(hashtag, list(iterator_for_tuples_of_location_and_hashtag_and_occurrence_time))
                        for hashtag, iterator_for_tuples_of_location_and_hashtag_and_occurrence_time in 
                            groupby(
                                sorted(tuples_of_location_and_hashtag_and_occurrence_time, key=itemgetter(1)),
                                key=itemgetter(1)
                            )
                       ]
                for hashtag, tuples_of_location_and_hashtag_and_occurrence_time in tuples_of_hashtag_and_tuples_of_location_and_hashtag_and_occurrence_time:
                    tuples_of_location_and_no_of_occurrences = [(location, len(list(iterator_of_locations)))
                             for location, iterator_of_locations in groupby(
                                     sorted(zip(*tuples_of_location_and_hashtag_and_occurrence_time)[0], key=itemgetter(0,1)),
                                     key=itemgetter(0,1)
                                     )
                             ]
                    locations, colors = zip(*tuples_of_location_and_no_of_occurrences)
#                    plotPointsOnWorldMap(locations, c=colors, cmap=matplotlib.cm.cool, lw = 0, alpha=1.0)
                    print timeUnitForPropagationForPrediction
                    print output_file(timeUnitForPropagationForPrediction, hashtag)
#                    plt.show()               
#                exit()
                del historicalTimeUnitsMap[timeUnitForPropagationForPrediction]; #del predictionTimeUnitsMap[timeUnitForActualPropagation]
            currentTime+=timeUnitDelta
    @staticmethod
    def run():
        PaperPlots.temp()
        pass
    
prediction_models = [
#                        PredictionModels.RANDOM , 
#                        PredictionModels.GREEDY, 
                        PredictionModels.SHARING_PROBABILITY, 
                        PredictionModels.TRANSMITTING_PROBABILITY,
#                        PredictionModels.COVERAGE_PROBABILITY, PredictionModels.SHARING_PROBABILITY_WITH_COVERAGE, PredictionModels.TRANSMITTING_PROBABILITY_WITH_COVERAGE,
                        PredictionModels.COVERAGE_DISTANCE, 
#                        PredictionModels.SHARING_PROBABILITY_WITH_COVERAGE_DISTANCE, PredictionModels.TRANSMITTING_PROBABILITY_WITH_COVERAGE_DISTANCE
                        ]

#prediction_models = [PredictionModels.COVERAGE_DISTANCE, PredictionModels.SHARING_PROBABILITY, PredictionModels.TRANSMITTING_PROBABILITY]
#plotAllData(prediction_models)
#getHashtagColors()
#plotRealData()
#plotCoverageDistance()

#GeneralAnalysis.grid_visualization()
#LearningAnalysis.run()
PaperPlots.run()
