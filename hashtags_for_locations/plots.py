'''
Created on Feb 27, 2012

@author: kykamath
'''
from operator import itemgetter
from analysis import iterateJsonFromFile
from itertools import groupby
from library.geo import getLocationFromLid, plotPointsOnWorldMap, getLatticeLid
from collections import defaultdict
import matplotlib.pyplot as plt
from library.classes import GeneralMethods
from models import loadLocationsList,\
    PredictionModels, Propagations, PREDICTION_MODEL_METHODS
from mr_analysis import LOCATION_ACCURACY


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

def plotLearningAnalysis():
#    file = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/testing/models/2011-09-01_2011-11-01/30_60/4/follow_the_leader_weights'
    file = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/testing/models/2011-09-01_2011-11-01/30_60/4/hedging_method_weights'
    final_map_from_location_to_map_from_model_to_weight = {}
    for data in iterateJsonFromFile(file):
        map_from_location_to_map_from_model_to_weight = data['location_weights']
        for location, map_from_model_to_weight in map_from_location_to_map_from_model_to_weight.iteritems():
            final_map_from_location_to_map_from_model_to_weight[location] = map_from_model_to_weight
    for location, map_from_model_to_weight in final_map_from_location_to_map_from_model_to_weight.iteritems():
        print location, map_from_model_to_weight

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
plotLearningAnalysis()

