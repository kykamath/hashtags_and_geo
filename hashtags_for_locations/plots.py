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
            getLatticeLid, plot_graph_clusters_on_world_map, isWithinBoundingBox,\
    plotPointsOnUSMap, getLattice
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
        hashtagsWithoutEndingWindowWithoutLatticeApproximationFile, \
        data_analysis_folder
from datetime import datetime
from library.stats import getOutliersRangeUsingIRQ, filter_outliers
import numpy as np
from hashtags_for_locations.models import loadSharingProbabilities,\
    EvaluationMetrics, CoverageModel, LOCATIONS_LIST, PredictionModels,\
    loadTransmittingProbabilities,\
    filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance
import networkx as nx
from library.graphs import clusterUsingAffinityPropagation
from scipy.stats import ks_2samp
from library.plotting import CurveFit, splineSmooth
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    locationsGraphFile = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/complete_prop/2011-05-01_2011-12-31/latticeGraph'
    tuples_of_location_and_tuples_of_neighbor_location_and_transmission_score_file = 'data/tuples_of_location_and_tuples_of_neighbor_location_and_transmission_score'
    SOURCE_COLOR = 'r'
    @staticmethod
    def grid_visualization():
        BIN_ACCURACY = 1.45
#        BIN_ACCURACY = 0.145
        map_from_location_bin_to_color = {}
        set_of_observed_location_ids = set()
        tuples_of_location_and_bin_color = []
        for count, data in enumerate(iterateJsonFromFile(hashtagsWithoutEndingWindowWithoutLatticeApproximationFile%('complete_prop', '2011-05-01', '2011-12-31'))):
            for location, time in data['oc']:
                location_id = getLatticeLid(location, LOCATION_ACCURACY)
                if location_id not in set_of_observed_location_ids:
                    set_of_observed_location_ids.add(location_id)
                    location_bin = getLatticeLid(location, BIN_ACCURACY)
                    if location_bin not in map_from_location_bin_to_color: map_from_location_bin_to_color[location_bin] = GeneralMethods.getRandomColor()
                    tuples_of_location_and_bin_color.append((location, map_from_location_bin_to_color[location_bin]))
            print count
#            if count==1000: break
        locations, colors = zip(*tuples_of_location_and_bin_color)
        plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=colors, lw = 0)
        plt.show()
        file_learning_analysis = './images/%s.png'%(GeneralMethods.get_method_id())
        FileIO.createDirectoryForFile(file_learning_analysis)
#        plt.savefig(file_learning_analysis)
    @staticmethod
    def write_transmission_scores_file():
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
        def get_occurrences_stats(occurrences1, occurrences2):
            no_of_occurrences_after_appearing_in_location, no_of_occurrences_before_appearing_in_location = 0., 0.
            occurrences1=sorted(occurrences1)
            occurrences2=sorted(occurrences2)
            no_of_total_occurrences_between_location_pair = len(occurrences1)*len(occurrences2)*1.
            for occurrence1 in occurrences1:
                for occurrence2 in occurrences2:
                    if occurrence1<occurrence2: no_of_occurrences_after_appearing_in_location+=1
                    elif occurrence1>occurrence2: no_of_occurrences_before_appearing_in_location+=1
            return no_of_occurrences_after_appearing_in_location, no_of_occurrences_before_appearing_in_location, no_of_total_occurrences_between_location_pair
        def get_transmission_score(no_of_occurrences_after_appearing_in_location, no_of_occurrences_before_appearing_in_location, no_of_total_occurrences_between_location_pair):
            return (no_of_occurrences_after_appearing_in_location - no_of_occurrences_before_appearing_in_location) / no_of_total_occurrences_between_location_pair
        GeneralMethods.runCommand('rm -rf %s'%GeneralAnalysis.tuples_of_location_and_tuples_of_neighbor_location_and_transmission_score_file)
        for line_count, location_object in enumerate(iterateJsonFromFile(GeneralAnalysis.locationsGraphFile)):
            print line_count
            tuples_of_neighbor_location_and_transmission_score = []
            map_from_hashtag_to_hashtag_weights = get_hashtag_weights(location_object['hashtags'])
            map_from_location_to_location_weights = get_location_weights(location_object['hashtags'], location_object['links'])
            for neighbor_location, map_from_hashtag_to_tuples_of_occurrences_and_time_range in location_object['links'].iteritems():
                transmission_scores = []
                for hashtag, (neighbor_location_occurrences, time_range) in map_from_hashtag_to_tuples_of_occurrences_and_time_range.iteritems():
                    if hashtag in location_object['hashtags']:
                        location_occurrences = location_object['hashtags'][hashtag][0]
                        (no_of_occurrences_after_appearing_in_location, \
                         no_of_occurrences_before_appearing_in_location, \
                         no_of_total_occurrences_between_location_pair)= get_occurrences_stats(location_occurrences, neighbor_location_occurrences)
                        transmission_scores.append(map_from_hashtag_to_hashtag_weights[hashtag]*get_transmission_score(no_of_occurrences_after_appearing_in_location, 
                                                                                                                       no_of_occurrences_before_appearing_in_location, 
                                                                                                                       no_of_total_occurrences_between_location_pair))
                mean_transmission_score = np.mean(transmission_scores)
                tuples_of_neighbor_location_and_transmission_score.append([neighbor_location, 
                                                                           map_from_location_to_location_weights[neighbor_location]*\
                                                                           mean_transmission_score])
            tuples_of_neighbor_location_and_transmission_score = sorted(tuples_of_neighbor_location_and_transmission_score,
                                                                        key=itemgetter(1))
            FileIO.writeToFileAsJson([location_object['id'], tuples_of_neighbor_location_and_transmission_score], 
                                     GeneralAnalysis.tuples_of_location_and_tuples_of_neighbor_location_and_transmission_score_file)
    @staticmethod
    def load_tuples_of_location_and_tuples_of_neighbor_location_and_transmission_score():
        return [(location, tuples_of_neighbor_location_and_transmission_score)
                 for location, tuples_of_neighbor_location_and_transmission_score in 
                 iterateJsonFromFile(GeneralAnalysis.tuples_of_location_and_tuples_of_neighbor_location_and_transmission_score_file)]
    @staticmethod
    def outgoing_and_incoming_locations_on_world_map():
        def plot_locations(source_location, tuples_of_location_and_transmission_score):
            source_location = getLocationFromLid(source_location.replace('_', ' '))
            if tuples_of_location_and_transmission_score:
                locations, transmission_scores = zip(*sorted(
                                                       tuples_of_location_and_transmission_score,
                                                       key=lambda (location, transmission_score): abs(transmission_score)
                                                       ))
                locations = [getLocationFromLid(location.replace('_', ' ')) for location in locations]
                transmission_scores = [abs(transmission_score) for transmission_score in transmission_scores]
                sc = plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=transmission_scores, cmap=matplotlib.cm.winter,  lw = 0)
                plt.colorbar(sc)
            plotPointsOnWorldMap([source_location], blueMarble=False, bkcolor='#CFCFCF', c=GeneralAnalysis.SOURCE_COLOR, lw = 0)
        tuples_of_location_and_tuples_of_neighbor_location_and_transmission_score = GeneralAnalysis.load_tuples_of_location_and_tuples_of_neighbor_location_and_transmission_score()
        for location_count, (location, tuples_of_neighbor_location_and_transmission_score) in enumerate(tuples_of_location_and_tuples_of_neighbor_location_and_transmission_score):
            output_file = data_analysis_folder%GeneralMethods.get_method_id()+'%s.png'%location
            print location_count, output_file
            tuples_of_outgoing_location_and_transmission_score = filter(lambda (neighbor_location, transmission_score): transmission_score>0, tuples_of_neighbor_location_and_transmission_score)
            tuples_of_incoming_location_and_transmission_score = filter(lambda (neighbor_location, transmission_score): transmission_score<0, tuples_of_neighbor_location_and_transmission_score)
            plt.subplot(211)
            plot_locations(location, tuples_of_outgoing_location_and_transmission_score)
            plt.title('Influences')
            plt.subplot(212)
            plot_locations(location, tuples_of_incoming_location_and_transmission_score)
            plt.title('Gets influenced by')
            FileIO.createDirectoryForFile(output_file)
            plt.savefig(output_file)
            plt.clf()
    @staticmethod
    def get_top_influencers(boundary):
        '''
        World
            London (Center), Washington D.C, New York (Brooklyn), London (South), Detroit
            Los Angeles, New York (Babylon), Atlanta, Sao Paulo, Miami 
        '''
        map_from_location_to_total_influence_score, set_of_locations = {}, set()
        tuples_of_location_and_tuples_of_neighbor_location_and_transmission_score = GeneralAnalysis.load_tuples_of_location_and_tuples_of_neighbor_location_and_transmission_score()
        for location, tuples_of_neighbor_location_and_transmission_score in tuples_of_location_and_tuples_of_neighbor_location_and_transmission_score:
            if isWithinBoundingBox(getLocationFromLid(location.replace('_', ' ')), boundary):
                set_of_locations.add(location)
                tuples_of_incoming_location_and_transmission_score = filter(lambda (neighbor_location, transmission_score): transmission_score<0, tuples_of_neighbor_location_and_transmission_score)
                for incoming_location, transmission_score in tuples_of_incoming_location_and_transmission_score:
                    if incoming_location not in map_from_location_to_total_influence_score: map_from_location_to_total_influence_score[incoming_location]=0.
                    map_from_location_to_total_influence_score[incoming_location]+=abs(transmission_score)
        no_of_locations = len(set_of_locations)
        tuples_of_location_and_mean_influence_scores = sorted([(location, total_influence_score/no_of_locations)
                                                             for location, total_influence_score in 
                                                             map_from_location_to_total_influence_score.iteritems()],
                                                         key=itemgetter(1), reverse=True)[:10]
        locations = zip(*tuples_of_location_and_mean_influence_scores)[0]
        locations = [getLocationFromLid(location.replace('_', ' ')) for location in locations]
        print locations
        plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c='r',  lw = 0)
        plt.show()
    @staticmethod
    def get_hashtags():
        set_of_hashtags = set()
        for line_count, location_object in enumerate(iterateJsonFromFile(GeneralAnalysis.locationsGraphFile)):
            print line_count
            [set_of_hashtags.add(hashtag) for hashtag in location_object['hashtags']]
        print len(set_of_hashtags)       
#    @staticmethod
#    def transmitting_sharing_relationships():
#        def load_incoming_and_outgoing_probabilities():
#            probabilities = defaultdict(dict)
#            for latticeObject in iterateJsonFromFile(GeneralAnalysis.locationsGraphFile):
#                latticeHashtagsSet = set(latticeObject['hashtags'])
#                probabilities[latticeObject['id']] = {'incoming': defaultdict(dict), 'outgoing': defaultdict(dict)}
#                for neighborLattice, neighborHashtags in latticeObject['links'].iteritems():
#                    neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags, findLag=False)
#                    incoming_hashtags = [k for k in neighborHashtags if k in latticeObject['hashtags'] and latticeObject['hashtags'][k][0]>neighborHashtags[k][0]]
#                    outgoing_hashtags = [k for k in neighborHashtags if k in latticeObject['hashtags'] and latticeObject['hashtags'][k][0]<neighborHashtags[k][0]]
#                    probabilities[latticeObject['id']]['incoming'][neighborLattice]=len(incoming_hashtags)/float(len(latticeHashtagsSet))
#                    probabilities[latticeObject['id']]['outgoing'][neighborLattice]=len(outgoing_hashtags)/float(len(latticeHashtagsSet))
#            return probabilities
#                
#        LOCATION_ACCURACY = 0.725
#        input_point = [40.762601,-73.97953]
#        input_point_lid = getLatticeLid(getLattice(input_point, LOCATION_ACCURACY), LOCATION_ACCURACY)
#        map_from_location_to_incoming_and_outgoing_probabilities = write_transmission_scores_file()
#        for location, incoming_and_outgoing_probabilities in map_from_location_to_incoming_and_outgoing_probabilities.iteritems():
#            for probability_type, map_from_neigboring_location_to_probabilities in incoming_and_outgoing_probabilities.iteritems():
#                print location, probability_type, sorted(map_from_neigboring_location_to_probabilities.iteritems(), key=itemgetter(1), reverse=True)
#            exit()
#            if location == input_point_lid:
#                tuples_of_location_and_probabilities = sorted(probabilities['neighborProbability'][location].iteritems(), key=itemgetter(1), reverse=True)
#                print tuples_of_location_and_probabilities
#        locations = []
#        for location in sharing_probabilities['neighborProbability']: 
#            if location==input_point_lid: locations.append(location)
#        locations = [getLocationFromLid(location.replace('_', ' ')) for location in locations]
#        plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c='r', lw = 0)
#        plt.show()
    @staticmethod
    def run():
#        GeneralAnalysis.grid_visualization()
#        GeneralAnalysis.write_transmission_scores_file()
#        GeneralAnalysis.outgoing_and_incoming_locations_on_world_map()

#        boundary = [[-90,-180], [90, 180]] # World
#        boundary = [[24.527135,-127.792969], [49.61071,-59.765625]] #USA
#        boundary = [[10.107706,-118.660469], [26.40009,-93.699531]] # Mexico
        boundary = [[-29.565473,-58.191719], [7.327985,-30.418282]] # Brazil
#        boundary = [[-16.6695,88.409841], [30.115057,119.698904]] #South East Asia
#        GeneralAnalysis.get_top_influencers(boundary)
        GeneralAnalysis.get_hashtags()
        
        
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
    hashtag_ditribution_on_world_map_file = 'data/hashtag_ditribution_on_world_map'
    map_from_hahstags_to_hashtag_properties = {
                                               'usopen' : {'color': '#0011FF'}, 
                                               'blackparentsquotes' : {'color': '#FF00EE'},
                                               'missuniverso' : {'color': '#00FF33'},
                                               }
    @staticmethod
    def get_tuples_of_hashtag_and_location_and_occurrence_time(): 
        tuples_of_hashtag_and_location_and_occurrence_time = [data for data in iterateJsonFromFile(PaperPlots.hashtag_ditribution_on_world_map_file)][0]['oc']
        return sorted(tuples_of_hashtag_and_location_and_occurrence_time, key=itemgetter(0))
    @staticmethod
    def hashtag_ditribution_on_world_map_by_time_units(generate_data=True):
        '''
        2011-09-04 13/00/00
        2011-09-12 19:00:00
        '''
        startTime, endTime, outputFolder = datetime(2011, 4, 1), datetime(2012, 1, 31), 'complete'
        time_unit_with_occurrences_file = timeUnitWithOccurrencesFile%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
        print 'Using file: ', time_unit_with_occurrences_file 
        if not generate_data:
            output_file_format = '/data/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/hashtag_ditribution_on_world_map_by_time_units/%s/%s.png'
            for data in iterateJsonFromFile(time_unit_with_occurrences_file):
                tuples_of_hashtag_and_location_and_occurrence_time = sorted(data['oc'], key=itemgetter(0))
                tuples_of_hashtag_and_locations = [ (hashtag, zip(*iterator_of_tuples_of_hashtag_and_location_and_occurrence_time)[1])
                                                    for hashtag, iterator_of_tuples_of_hashtag_and_location_and_occurrence_time in 
                                                    groupby(tuples_of_hashtag_and_location_and_occurrence_time, key=itemgetter(0))
                                                ]
                for hashtag, locations in tuples_of_hashtag_and_locations:
                    tuples_of_location_and_occurrences_count = [(getLocationFromLid(location.replace('_', ' ')), len(list(itertor_of_locations))) for location, itertor_of_locations in groupby(sorted(locations))]
                    if len(tuples_of_location_and_occurrences_count)>25:
                        locations, colors = zip(*sorted(tuples_of_location_and_occurrences_count, key=itemgetter(1)))
                        sc = plotPointsOnWorldMap(locations, c=colors, cmap=matplotlib.cm.cool, lw = 0, alpha=1.0)
                        output_file = output_file_format%(datetime.fromtimestamp(data['tu']), unicode(hashtag).encode('utf-8'))
                        print output_file
                        FileIO.createDirectoryForFile(output_file)
                        plt.colorbar(sc)
                        plt.savefig(output_file)
    #                    plt.show()               
                        plt.clf()
        else:
            time_unit_to_output = '2011-09-12 19:00:00'
            for data in iterateJsonFromFile(time_unit_with_occurrences_file):
                if time_unit_to_output==str(datetime.fromtimestamp(data['tu'])):
                    FileIO.writeToFileAsJson(data, PaperPlots.hashtag_ditribution_on_world_map_file)
    @staticmethod
    def raw_data_on_world_map():
        tuples_of_hashtag_and_location_and_occurrence_time = PaperPlots.get_tuples_of_hashtag_and_location_and_occurrence_time()
        tuples_of_hashtag_and_locations = [ (hashtag, zip(*iterator_of_tuples_of_hashtag_and_location_and_occurrence_time)[1])
                                                    for hashtag, iterator_of_tuples_of_hashtag_and_location_and_occurrence_time in 
                                                        groupby(tuples_of_hashtag_and_location_and_occurrence_time, key=itemgetter(0))
                                                    if hashtag in PaperPlots.map_from_hahstags_to_hashtag_properties
                                                ]
        for hashtag, locations in tuples_of_hashtag_and_locations:
            tuples_of_location_and_occurrences_count = [(getLocationFromLid(location.replace('_', ' ')), len(list(itertor_of_locations))) for location, itertor_of_locations in groupby(sorted(locations))]
            locations, colors = zip(*sorted(tuples_of_location_and_occurrences_count, key=itemgetter(1)))
            ax=plt.subplot(111)
            sc = plotPointsOnWorldMap(locations, c=colors, bkcolor='#CFCFCF', cmap=matplotlib.cm.winter, lw = 0, alpha=1.0)
            output_file = 'images/%s_%s.png'%(GeneralMethods.get_method_id(), hashtag)
            print output_file
            FileIO.createDirectoryForFile(output_file)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(sc, cax)
            plt.savefig(output_file)
#                plt.show()               
            plt.clf()
    @staticmethod
    def coverage_metrics_on_world_map():
        LOCATIONS_LIST = loadLocationsList()
        for coverage_metric_method, coverage_metric_id in [(CoverageModel.spreadProbability, PredictionModels.COVERAGE_PROBABILITY), (CoverageModel.spreadDistance, PredictionModels.COVERAGE_DISTANCE)]:
            tuples_of_hashtag_and_location_and_occurrence_time = PaperPlots.get_tuples_of_hashtag_and_location_and_occurrence_time()
            tuples_of_hashtag_and_locations = [ (hashtag, zip(*iterator_of_tuples_of_hashtag_and_location_and_occurrence_time)[1])
                                                        for hashtag, iterator_of_tuples_of_hashtag_and_location_and_occurrence_time in 
                                                            groupby(tuples_of_hashtag_and_location_and_occurrence_time, key=itemgetter(0))
                                                        if hashtag in PaperPlots.map_from_hahstags_to_hashtag_properties
                                                    ]
            map_from_hashtag_to_map_from_location_to_coverage_metric = defaultdict(dict)
            for hashtag, locations in tuples_of_hashtag_and_locations:
                locations = [getLocationFromLid(location.replace('_', ' ')) for location in locations]
                map_from_hashtag_to_map_from_location_to_coverage_metric[hashtag] = coverage_metric_method(locations, LOCATIONS_LIST)
            map_from_location_to_tuple_of_coverage_metric_and_hashtag = {}
            for hashtag, map_from_location_to_coverage_metric in map_from_hashtag_to_map_from_location_to_coverage_metric.iteritems():
                for location, coverage_metric in map_from_location_to_coverage_metric.iteritems():
                    if location not in map_from_location_to_tuple_of_coverage_metric_and_hashtag: map_from_location_to_tuple_of_coverage_metric_and_hashtag[location] = [coverage_metric, hashtag]
                    elif map_from_location_to_tuple_of_coverage_metric_and_hashtag[location][0]<coverage_metric: map_from_location_to_tuple_of_coverage_metric_and_hashtag[location] = [coverage_metric, hashtag]
            tuples_of_location_and_color = [(getLocationFromLid(location.replace('_', ' ')), PaperPlots.map_from_hahstags_to_hashtag_properties[hashtag]['color']) 
                                      for location, (coverage_metric, hashtag) in 
                                      map_from_location_to_tuple_of_coverage_metric_and_hashtag.iteritems()
                                  ]
            locations, colors = zip(*sorted(tuples_of_location_and_color, key=itemgetter(1), reverse=True))
#            plotPointsOnUSMap(locations, c=colors, lw = 0, s=80, alpha=1.0)
            plotPointsOnWorldMap(locations, c=colors, bkcolor='#CFCFCF', lw = 0, alpha=1.0)
            for hashtag, hashtag_properties in sorted(PaperPlots.map_from_hahstags_to_hashtag_properties.iteritems(), key=itemgetter(0)): plt.scatter(0,0, label='%s'%hashtag, color=hashtag_properties['color'])
            output_file = 'images/%s_%s.png'%(GeneralMethods.get_method_id(), coverage_metric_id)
            print output_file
            plt.legend(loc=3, ncol=3, mode="expand",)
            FileIO.createDirectoryForFile(output_file)
            plt.savefig(output_file)
#            plt.show()
            plt.clf()
#    @staticmethod
#    def temp():
#        conf = {'noOfTargetHashtags': 1}
#        propogations = Propagations(None, None)
#        propogations.update(PaperPlots.get_tuples_of_hashtag_and_location_and_occurrence_time())
#        probabilities = loadTransmittingProbabilities()
#        print PredictionModels._hashtags_by_location_probabilities(propogations, probabilities, **conf)
        
    @staticmethod
    def run():
#        PaperPlots.hashtag_ditribution_on_world_map_by_time_units()
#        PaperPlots.raw_data_on_world_map()    
        PaperPlots.coverage_metrics_on_world_map()
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

GeneralAnalysis.run()
#LearningAnalysis.run()
#PaperPlots.run()
