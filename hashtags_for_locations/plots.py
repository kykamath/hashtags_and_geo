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
import random, matplotlib, inspect
from library.file_io import FileIO
from models import ModelSelectionHistory
from settings import analysisFolder, timeUnitWithOccurrencesFile, \
        PARTIAL_WORLD_BOUNDARY
from datetime import datetime
from library.stats import getOutliersRangeUsingIRQ
import numpy as np
from hashtags_for_locations.models import loadSharingProbabilities
import networkx as nx
from library.graphs import clusterUsingAffinityPropagation
from scipy.stats import ks_2samp
from library.plotting import CurveFit, splineSmooth

ALL_LOCATIONS = 'all_locations'
MAP_FROM_MODEL_TO_COLOR = dict([
                                (PredictionModels.COVERAGE_DISTANCE, 'b'), (PredictionModels.COVERAGE_PROBABILITY, 'm'), (PredictionModels.SHARING_PROBABILITY, 'r'), (PredictionModels.TRANSMITTING_PROBABILITY, 'k'),
                                (ModelSelectionHistory.FOLLOW_THE_LEADER, '#FF0A0A'), (ModelSelectionHistory.HEDGING_METHOD, '#9661FF'),
                                (PredictionModels.COMMUNITY_AFFINITY, '#436DFC'), (PredictionModels.SPATIAL, '#F15CFF'), (ALL_LOCATIONS, '#FFB44A')
                                ])
MAP_FROM_MODEL_TO_MODEL_TYPE = dict([
                                     (PredictionModels.SHARING_PROBABILITY, PredictionModels.COMMUNITY_AFFINITY),
                                     (PredictionModels.TRANSMITTING_PROBABILITY, PredictionModels.COMMUNITY_AFFINITY),
                                     (PredictionModels.COVERAGE_DISTANCE, PredictionModels.SPATIAL),
                                     (PredictionModels.COVERAGE_PROBABILITY, PredictionModels.SPATIAL),
                                     ])


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
            
            
#            #Plot coverge distance
#            plt.figure()
##            hashtag_and_coverage_distance = [(hashtag, CoverageModel.spreadDistance(points)) for hashtag, points in hashtag_and_occurrence_locations]
#            hashtag_and_coverage_distance = propagation.getCoverageDistances().items()
#            location_to_hashtag_and_coverage_distance_value_map = defaultdict(list)
#            for hashtag, coverage_distance in hashtag_and_coverage_distance:
#                for location, coverage_distance_value in coverage_distance.iteritems(): location_to_hashtag_and_coverage_distance_value_map[location].append([hashtag, coverage_distance_value])
#            location_and_colors = []
#            for location, hashtag_and_coverage_distance_value in location_to_hashtag_and_coverage_distance_value_map.iteritems():
#                hashtag, coverage_distance_value = max(hashtag_and_coverage_distance_value, key=itemgetter(1))
#                if hashtag in map_from_hashtag_to_color: location_and_colors.append([getLocationFromLid(location.replace('_', ' ')), map_from_hashtag_to_color[hashtag] ])
#            points, colors = zip(*location_and_colors)
#            plotPointsOnWorldMap(points, blueMarble=False, bkcolor='#CFCFCF', c=colors, lw = 0)

#            map_from_location_to_list_of_predicted_hashtags = PredictionModels.coverage_distance(propagation, **conf)
#            list_of_tuple_of_location_and_color = [(getLocationFromLid(location.replace('_', ' ')), map_from_hashtag_to_color[hashtags[0]]) for location, hashtags in map_from_location_to_list_of_predicted_hashtags.iteritems() if hashtags and hashtags[0] in map_from_hashtag_to_color]
#            locations, colors = zip(*list_of_tuple_of_location_and_color)
#            plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=colors, lw = 0)
#            plt.title(PredictionModels.COVERAGE_DISTANCE)
            
#            #Plot sharing probability
#            plt.figure()
##            SHARING_PROBABILITIES = loadSharingProbabilities()
##            map_from_location_to_list_of_predicted_hashtags = PredictionModels._hashtags_by_location_probabilities(propagation, SHARING_PROBABILITIES, **conf)
#            map_from_location_to_list_of_predicted_hashtags = PredictionModels.sharing_probability(propagation, **conf)
#            list_of_tuple_of_location_and_color = [(getLocationFromLid(location.replace('_', ' ')), map_from_hashtag_to_color[hashtags[0]]) for location, hashtags in map_from_location_to_list_of_predicted_hashtags.iteritems() if hashtags and hashtags[0] in map_from_hashtag_to_color]
#            locations, colors = zip(*list_of_tuple_of_location_and_color)
#            plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=colors, lw = 0)
#            plt.title(PredictionModels.SHARING_PROBABILITY)
#            
#            #Plot transmitting probability
#            plt.figure()
##            TRANSMITTING_PROBABILITIES = loadTransmittingProbabilities()
##            map_from_location_to_list_of_predicted_hashtags = PredictionModels._hashtags_by_location_probabilities(propagation, TRANSMITTING_PROBABILITIES, **conf)
#            map_from_location_to_list_of_predicted_hashtags = PredictionModels.transmitting_probability(propagation, **conf)
#            list_of_tuple_of_location_and_color = [(getLocationFromLid(location.replace('_', ' ')), map_from_hashtag_to_color[hashtags[0]]) for location, hashtags in map_from_location_to_list_of_predicted_hashtags.iteritems() if hashtags and hashtags[0] in map_from_hashtag_to_color]
#            locations, colors = zip(*list_of_tuple_of_location_and_color)
#            plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=colors, lw = 0)
#            plt.title(PredictionModels.TRANSMITTING_PROBABILITY)
            
#            plt.show()

def plot_model_distribution_on_world_map(learning_type, no_of_hashtags, generate_data=True):
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
            plt.savefig('images/learning_analysis/%s.png'%model)
            plt.clf()
            
def plot_location_size_to_model_correlation(learning_type):
    ACCURACY = 100
    weights_analysis_file = analysisFolder%'learning_analysis'+'/%s_weights_analysis'%(learning_type)
    tuples_of_location_and_best_model = [tuple_of_location_and_best_model for tuple_of_location_and_best_model in FileIO.iterateJsonFromFile(weights_analysis_file)]
    map_from_location_to_best_model = dict(tuples_of_location_and_best_model)
    
    startTime, endTime, outputFolder = datetime(2011, 9, 1), datetime(2011, 11, 1), 'testing'
    input_file = timeUnitWithOccurrencesFile%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
    map_from_location_to_no_of_occurrences_at_location = defaultdict(float)
    for time_unit_object in iterateJsonFromFile(input_file):
        for (_, location, _) in time_unit_object['oc']: 
            if location in map_from_location_to_best_model: map_from_location_to_no_of_occurrences_at_location[location]+=1
    map_from_model_to_tuples_of_location_and_no_of_occurrences_at_location = dict([(model, [(location, map_from_location_to_no_of_occurrences_at_location[location]) for location in zip(*iterator_of_tuples_of_location_and_models)[0]]) 
                                                                                   for model, iterator_of_tuples_of_location_and_models in 
                                                                                   groupby(
                                                                                          sorted(tuples_of_location_and_best_model, key=itemgetter(1)),
                                                                                          key=itemgetter(1)
                                                                                          )
                                                                               ])
    map_from_model_to_tuples_of_location_and_no_of_occurrences_at_location[ALL_LOCATIONS] = map_from_location_to_no_of_occurrences_at_location.items()
#    for model, tuples_of_location_and_no_of_occurrences_at_location in map_from_model_to_tuples_of_location_and_no_of_occurrences_at_location.items()[:]: print model, len(tuples_of_location_and_no_of_occurrences_at_location)
    for model, tuples_of_location_and_no_of_occurrences_at_location in map_from_model_to_tuples_of_location_and_no_of_occurrences_at_location.items()[:]:
        _, upper_range_for_no_of_occurrences_at_location = getOutliersRangeUsingIRQ(zip(*tuples_of_location_and_no_of_occurrences_at_location)[1])
        map_from_model_to_tuples_of_location_and_no_of_occurrences_at_location[model] = filter(
                                                                                               lambda (location, no_of_occurrences_at_location): no_of_occurrences_at_location < upper_range_for_no_of_occurrences_at_location, 
                                                                                               tuples_of_location_and_no_of_occurrences_at_location)
#    print '**********'
#    for model, tuples_of_location_and_no_of_occurrences_at_location in map_from_model_to_tuples_of_location_and_no_of_occurrences_at_location.items()[:]: print model, len(tuples_of_location_and_no_of_occurrences_at_location)
#    exit()
    tuples_of_model_and_tuples_of_location_and_no_of_occurrences_at_location = map_from_model_to_tuples_of_location_and_no_of_occurrences_at_location.items()
    for model, tuples_of_location_and_no_of_occurrences_at_location in tuples_of_model_and_tuples_of_location_and_no_of_occurrences_at_location:
        print model, ks_2samp(zip(*map_from_model_to_tuples_of_location_and_no_of_occurrences_at_location[ALL_LOCATIONS])[1], list(zip(*tuples_of_location_and_no_of_occurrences_at_location)[1]))
        
    map_from_model_to_map_from_population_to_population_distribution = defaultdict(dict)
    for model, tuples_of_location_and_no_of_occurrences_at_location in tuples_of_model_and_tuples_of_location_and_no_of_occurrences_at_location:
        list_of_no_of_occurrences_at_location = zip(*tuples_of_location_and_no_of_occurrences_at_location)[1]
        for population in list_of_no_of_occurrences_at_location: 
            population = int(population)/ACCURACY*ACCURACY + ACCURACY
            if population not in map_from_model_to_map_from_population_to_population_distribution[model]:
                map_from_model_to_map_from_population_to_population_distribution[model][population]=0
            map_from_model_to_map_from_population_to_population_distribution[model][population]+=1
    for model, map_from_population_to_population_distribution in map_from_model_to_map_from_population_to_population_distribution.iteritems():
#        dataX = filter(lambda x: x<1000, sorted(map_from_population_to_population_distribution))
        dataX = sorted([x for x in map_from_population_to_population_distribution if map_from_population_to_population_distribution[x]>10])
        total_locations = float(sum(map_from_population_to_population_distribution[x] for x in dataX))
        dataY = [map_from_population_to_population_distribution[x]/total_locations for x in dataX]
        print model
        print dataX
        print [map_from_population_to_population_distribution[x] for x in dataX]
        print dataY
        parameters_after_fitting = CurveFit.getParamsAfterFittingData(dataX, dataY, CurveFit.decreasingExponentialFunction, [0., 0.])
        print CurveFit.getYValues(CurveFit.decreasingExponentialFunction, parameters_after_fitting, range(ACCURACY, 2400/ACCURACY*ACCURACY))
        plt.scatter(dataX, dataY, color=MAP_FROM_MODEL_TO_COLOR[model], label=model, lw=2)
        plt.loglog(range(ACCURACY, 2400/ACCURACY*ACCURACY), CurveFit.getYValues(CurveFit.decreasingExponentialFunction, parameters_after_fitting, range(ACCURACY, 2400/ACCURACY*ACCURACY)), color=MAP_FROM_MODEL_TO_COLOR[model])
#        plt.loglog(dataX[0], dataY[0])
    
    plt.legend()
#    plt.xlim(xmin=0.0)
#    plt.ylim(ymin=-0.4, ymax=0.8)
    plt.show()

def plot_model_learning_graphs(learning_type):
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
#    ############
#
#    map_from_model_type_to_locations = defaultdict(list)
#    for model, locations in tuples_of_model_and_locations: map_from_model_type_to_locations[map_from_model_to_model_type[model]]+=locations
#    tuples_of_model_and_locations = map_from_model_type_to_locations.iteritems()
#    #############
    
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
#        plt.title(model + ' (%s)'%no_of_clusters )
#        plt.show()
        plt.savefig('images/model_graph/%s.png'%model)
        plt.clf()

def follow_the_leader_method(map_from_model_to_weight): return min(map_from_model_to_weight.iteritems(), key=itemgetter(1))[0]
def hedging_method(map_from_model_to_weight):
    total_weight = sum(map_from_model_to_weight.values())
    for model in map_from_model_to_weight.keys(): map_from_model_to_weight[model]/=total_weight 
    tuple_of_id_model_and_cumulative_losses = [(id, model, cumulative_loss) for id, (model, cumulative_loss) in enumerate(map_from_model_to_weight.iteritems())]
    selected_id = GeneralMethods.weightedChoice(zip(*tuple_of_id_model_and_cumulative_losses)[2])
    return filter(lambda (id, model, _): id==selected_id, tuple_of_id_model_and_cumulative_losses)[0][1]
    
class LearningAnalysis():
    MAP_FROM_LEARNING_TYPE_TO_MODEL_SELECION_METHOD = dict([(ModelSelectionHistory.FOLLOW_THE_LEADER, follow_the_leader_method), (ModelSelectionHistory.HEDGING_METHOD, hedging_method)])
#    @staticmethod
#    def _get_location_learning_times(input_weight_file):
#        def get_final_model_change((reduced_time_unit, reduced_model), (current_time_unit, current_model)): 
#            if reduced_model!=current_model: return (current_time_unit, current_model)
#            else: return (reduced_time_unit, reduced_model)
#        map_from_location_to_tuples_of_time_unit_and_model_selected = defaultdict(list)
#        epoch_first_time_unit, tuples_of_location_and_last_time_unit_and_last_model_selected = None, []
#        for data in iterateJsonFromFile(input_weight_file):
#            map_from_location_to_map_from_model_to_weight = data['location_weights']
#            epoch_time_unit = data['tu']
#            for location, map_from_model_to_weight in map_from_location_to_map_from_model_to_weight.iteritems(): 
#                map_from_location_to_tuples_of_time_unit_and_model_selected[location].append([epoch_time_unit, min(map_from_model_to_weight.iteritems(), key=itemgetter(1))[0]])
#            if not epoch_first_time_unit: epoch_first_time_unit = epoch_time_unit
#        for location, tuples_of_time_unit_and_model_selected in map_from_location_to_tuples_of_time_unit_and_model_selected.iteritems():
#            last_time_unit, last_model_selected = reduce(get_final_model_change, tuples_of_time_unit_and_model_selected)
#            tuples_of_location_and_last_time_unit_and_last_model_selected.append([location, last_time_unit, last_model_selected])
#        return epoch_first_time_unit, tuples_of_location_and_last_time_unit_and_last_model_selected
#    @staticmethod
#    def plot_model_learning_time_series(learning_type, no_of_hashtags):
#        input_weight_file = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/testing/models/2011-09-01_2011-11-01/30_60/%s/%s_weights'%(no_of_hashtags, learning_type)
#        epoch_first_time_unit, tuples_of_location_and_last_time_unit_and_last_model_selected = LearningAnalysis._get_location_learning_times(input_weight_file)
#        total_locations = float(len(tuples_of_location_and_last_time_unit_and_last_model_selected))
#        tuples_of_time_unit_and_percentage_of_locations = [(time_unit, len(list(iterator_of_tuples_of_location_and_last_time_unit_and_last_model_selected)))
#                                                               for time_unit, iterator_of_tuples_of_location_and_last_time_unit_and_last_model_selected in
#                                                                   groupby(
#                                                                       sorted(tuples_of_location_and_last_time_unit_and_last_model_selected, key=itemgetter(1)),
#                                                                       key=itemgetter(1)
#                                                                   )
#                                                           ]
#        tuples_of_time_unit_and_cumulative_of_percentage_of_locations = []
#        cumulative_of_percentage_of_locations = 0.0
#        for time_unit, percentage_of_locations in sorted(tuples_of_time_unit_and_percentage_of_locations, key=itemgetter(0)):
#            cumulative_of_percentage_of_locations+=percentage_of_locations
#            tuples_of_time_unit_and_cumulative_of_percentage_of_locations.append((time_unit, cumulative_of_percentage_of_locations/total_locations))
#        dataX, dataY = zip(*sorted(tuples_of_time_unit_and_cumulative_of_percentage_of_locations, key=itemgetter(0)))
#    #    newDataX, dataY = splineSmooth(dataX, dataY)
#        plt.plot([(x-epoch_first_time_unit)/(60*60) for x in dataX], dataY)
#        plt.xlim(xmin = epoch_first_time_unit-epoch_first_time_unit)
#        plt.ylim(ymin=0, ymax=1.0)
#        plt.show()
#    @staticmethod
#    def plot_model_learning_time_on_map(learning_type, no_of_hashtags):
#        input_weight_file = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/testing/models/2011-09-01_2011-11-01/30_60/%s/%s_weights'%(no_of_hashtags, learning_type)
#        epoch_first_time_unit, tuples_of_location_and_last_time_unit_and_last_model_selected = LearningAnalysis._get_location_learning_times(input_weight_file) 
#        tuples_of_location_and_learning_time = [(location, (last_time_unit-epoch_first_time_unit)/(60*60))
#                                                for location, last_time_unit, _ in tuples_of_location_and_last_time_unit_and_last_model_selected
#                                                ]
#        tuples_of_locations_and_colors = [(getLocationFromLid(location.replace('_', ' ')), learning_time) for location, learning_time in tuples_of_location_and_learning_time]
#        locations, colors = zip(*sorted(tuples_of_locations_and_colors, key=itemgetter(1)))
#        plt.subplot(111)
#        sc = plotPointsOnWorldMap(locations, c=colors, cmap=matplotlib.cm.autumn, lw = 0, alpha=1.0)
#        plt.colorbar(sc)
#        plt.show()
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
            map_from_location_to_map_from_model_to_weight = data['location_weights']
            ep_time_unit = data['tu']
            set_of_ep_time_units.add(ep_time_unit)
            for location, map_from_model_to_weight in map_from_location_to_map_from_model_to_weight.iteritems():
                selected_model = MAP_FROM_MODEL_TO_MODEL_TYPE[LearningAnalysis.MAP_FROM_LEARNING_TYPE_TO_MODEL_SELECION_METHOD[learning_type](map_from_model_to_weight)]
                map_from_location_to_tuples_of_ep_time_unit_and_selected_model[location].append([ep_time_unit, selected_model])
            total_no_of_time_units = len(set_of_ep_time_units)
            for location, tuples_of_ep_time_unit_and_selected_model in map_from_location_to_tuples_of_ep_time_unit_and_selected_model.iteritems():
                non_flips_for_location, _ = reduce(count_non_flips, tuples_of_ep_time_unit_and_selected_model, (0.0, None))
                tuples_of_location_and_flipping_ratio.append([location, 1.0-(non_flips_for_location/total_no_of_time_units) ])
        return tuples_of_location_and_flipping_ratio
    @staticmethod
    def learner_flipping_time_series(learning_types, no_of_hashtags):
        for learning_type in learning_types:
            input_weight_file = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/testing/models/2011-09-01_2011-11-01/30_60/%s/%s_weights'%(no_of_hashtags, learning_type)
            map_from_ep_time_unit_to_no_of_locations_that_didnt_flip = {}
            map_from_location_to_previously_selected_model = {}
            for data in iterateJsonFromFile(input_weight_file):
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
            plt.plot([(x-ep_first_time_unit)/(60*60) for x in x_data], y_data, c=MAP_FROM_MODEL_TO_COLOR[learning_type], label=learning_type, lw=2)
        plt.legend()
#        plt.show()
        file_learning_analysis = './images/%s.png'%GeneralMethods.get_method_id()
        FileIO.createDirectoryForFile(file_learning_analysis)
        plt.savefig(file_learning_analysis)
        plt.clf()
    @staticmethod
    def flipping_ratio_on_world_map(learning_type, no_of_hashtags):
        tuples_of_location_and_flipping_ratio = LearningAnalysis._get_flipping_ratio_for_all_locations(learning_type, no_of_hashtags)
        for x, y in tuples_of_location_and_flipping_ratio:
            print x, y
        pass
    @staticmethod
    def run():
        no_of_hashtags = 4
#        LearningAnalysis.learner_flipping_time_series([ModelSelectionHistory.FOLLOW_THE_LEADER, ModelSelectionHistory.HEDGING_METHOD], no_of_hashtags)
        LearningAnalysis.flipping_ratio_on_world_map(ModelSelectionHistory.FOLLOW_THE_LEADER, no_of_hashtags)
            
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

#plot_model_distribution_on_world_map(learning_type=ModelSelectionHistory.FOLLOW_THE_LEADER, generate_data=False)
#plot_location_size_to_model_correlation(learning_type=ModelSelectionHistory.FOLLOW_THE_LEADER)
#plot_model_learning_graphs(learning_type=ModelSelectionHistory.FOLLOW_THE_LEADER)

LearningAnalysis.run()
