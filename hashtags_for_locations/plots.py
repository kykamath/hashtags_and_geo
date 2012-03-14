'''
Created on Feb 27, 2012

@author: kykamath
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
import random
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

ALL_LOCATIONS = 'all_locations'
MAP_FROM_MODEL_TO_COLOR = dict([
                                (PredictionModels.COVERAGE_DISTANCE, 'b'), (PredictionModels.COVERAGE_PROBABILITY, 'm'), (PredictionModels.SHARING_PROBABILITY, 'r'), (PredictionModels.TRANSMITTING_PROBABILITY, 'k'),
                                (PredictionModels.COMMUNITY_AFFINITY, '#436DFC'), (PredictionModels.SPATIAL, '#F15CFF'), (ALL_LOCATIONS, '#FFB44A')
                                ])
MAP_FROM_MODEL_TO_MODEL_TYPE = dict([
                                     (PredictionModels.SHARING_PROBABILITY, PredictionModels.COMMUNITY_AFFINITY),
                                     (PredictionModels.TRANSMITTING_PROBABILITY, PredictionModels.COMMUNITY_AFFINITY),
                                     (PredictionModels.COVERAGE_DISTANCE, PredictionModels.SPATIAL),
                                     (PredictionModels.COVERAGE_PROBABILITY, PredictionModels.SPATIAL),
                                     ])


def getHashtagColors(hashtag_and_occurrence_locations):
#        hashtag_and_points = [(h, map(lambda lid: getLocationFromLid(lid.replace('_', ' ')), zip(*occs)[1])) for h, occs in groupby(sorted(data['oc'], key=itemgetter(0)), key=itemgetter(0))]
#        print zip(*sorted([(hashtag, len(points)) for hashtag, points in hashtag_and_points if len(points)>25], key=itemgetter(1), reverse=True))[0]
#        return dict([
#                     ('replace1wordinamoviewithgrind', 'g'), 
#                     ('palmeirascampeaomundial51', 'm'), 
#                     ('11million', 'y'), 
#                     ('prayfornorway', 'k'), 
#                     ('happybirthdayselena', 'r'), 
#                     ])
#    cmaps = [matplotlib.cm.Blues, matplotlib.cm.Purples, matplotlib.cm.gist_yarg]
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

def plot_model_distribution_on_world_map(learning_type, generate_data=True):
    weights_analysis_file = analysisFolder%'learning_analysis'+'/%s_weights_analysis'%(learning_type)
    if generate_data:
        GeneralMethods.runCommand('rm -rf %s'%weights_analysis_file)
        input_weight_file = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/testing/models/2011-09-01_2011-11-01/30_60/4/%s_weights'%learning_type
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
    ACCURACY = 500
    weights_analysis_file = analysisFolder%'learning_analysis'+'/%s_weights_analysis'%(learning_type)
    tuples_of_location_and_best_model = [tuple_of_location_and_best_model for tuple_of_location_and_best_model in FileIO.iterateJsonFromFile(weights_analysis_file)]
    map_from_location_to_best_model = dict(tuples_of_location_and_best_model)
    
    startTime, endTime, outputFolder = datetime(2011, 9, 1), datetime(2011, 11, 1), 'testing'
    input_file = timeUnitWithOccurrencesFile%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
    map_from_location_to_no_of_occurrences_at_location = defaultdict(float)
    for time_unit_object in iterateJsonFromFile(input_file):
        for (_, location, _) in time_unit_object['oc']: 
            if location in map_from_location_to_best_model: map_from_location_to_no_of_occurrences_at_location[location]+=1
    tuples_of_model_and_tuples_of_location_and_no_of_occurrences_at_location = [(model, [(location, map_from_location_to_no_of_occurrences_at_location[location]) for location in zip(*iterator_of_tuples_of_location_and_models)[0]]) 
                                                                                   for model, iterator_of_tuples_of_location_and_models in 
                                                                                   groupby(
                                                                                          sorted(tuples_of_location_and_best_model, key=itemgetter(1)),
                                                                                          key=itemgetter(1)
                                                                                          )
                                                                               ]

    for model, tuples_of_location_and_no_of_occurrences_at_location in tuples_of_model_and_tuples_of_location_and_no_of_occurrences_at_location:
        print model, ks_2samp(map_from_location_to_no_of_occurrences_at_location.values(), list(zip(*tuples_of_location_and_no_of_occurrences_at_location)[1]))
        
    tuples_of_model_and_tuples_of_location_and_no_of_occurrences_at_location.append((ALL_LOCATIONS, map_from_location_to_no_of_occurrences_at_location.items()))
    map_from_model_to_map_from_population_to_population_distribution = defaultdict(dict)
    for model, tuples_of_location_and_no_of_occurrences_at_location in tuples_of_model_and_tuples_of_location_and_no_of_occurrences_at_location:
        list_of_no_of_occurrences_at_location = zip(*tuples_of_location_and_no_of_occurrences_at_location)[1]
        for population in list_of_no_of_occurrences_at_location: 
            population = int(population)/ACCURACY*ACCURACY + ACCURACY
            if population not in map_from_model_to_map_from_population_to_population_distribution[model]:
                map_from_model_to_map_from_population_to_population_distribution[model][population]=0
            map_from_model_to_map_from_population_to_population_distribution[model][population]+=1
    for model, map_from_population_to_population_distribution in map_from_model_to_map_from_population_to_population_distribution.iteritems():
        total_locations = float(sum(map_from_population_to_population_distribution.values()))
        dataX = sorted(map_from_population_to_population_distribution)
        dataY = [map_from_population_to_population_distribution[x]/total_locations for x in dataX]
        print model
        print dataX
        print dataY
        plt.loglog(dataX, dataY, color=MAP_FROM_MODEL_TO_COLOR[model], label=model, lw=2)
    
    plt.legend()
#    plt.xlim(xmin=0.0)
#    plt.ylim(ymin=-0.4, ymax=0.8)
    plt.show()

def temp(learning_type):
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
#            print cluster_id, len(locations_in_cluster)
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

#######################
        no_of_clusters, _ = plot_graph_clusters_on_world_map1(graph_of_locations)
#        plt.title(model + ' (%s)'%no_of_clusters )
#        plt.show()
        plt.savefig('images/model_graph/%s.png'%model)
        plt.clf()
#######################

#        no_of_clusters, tuples_of_location_and_cluster_id = clusterUsingAffinityPropagation(graph_of_locations)
#        tuples_of_cluster_id_and_locations_in_cluster = [(cluster_id, zip(*iterator_of_tuples_of_location_and_cluster_id)[0]) 
#                                                        for cluster_id, iterator_of_tuples_of_location_and_cluster_id in 
#                                                            groupby(
#                                                                    sorted(tuples_of_location_and_cluster_id, key=itemgetter(1)), 
#                                                                    key=itemgetter(1)
#                                                                    )
#                                                     ]
        
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

plot_model_distribution_on_world_map(learning_type=ModelSelectionHistory.FOLLOW_THE_LEADER, generate_data=False)
#plot_location_size_to_model_correlation(learning_type=ModelSelectionHistory.FOLLOW_THE_LEADER)
#temp(learning_type=ModelSelectionHistory.FOLLOW_THE_LEADER)

