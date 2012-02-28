'''
Created on Feb 27, 2012

@author: kykamath
'''
from operator import itemgetter
from hashtags_for_locations.analysis import iterateJsonFromFile
from itertools import groupby
from library.geo import getLocationFromLid, plotPointsOnWorldMap
from collections import defaultdict
import matplotlib.pyplot as plt
from library.classes import GeneralMethods
from hashtags_for_locations.models import CoverageModel, loadLocationsList


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
    return dict([(hashtag, GeneralMethods.getRandomColor()) for hashtag, occurrence_locations in hashtag_and_occurrence_locations if len(occurrence_locations)>0])

def plotAllData():
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
            #Plot real data.
            tuple_of_location_and__tuple_of_hashtag_and_no_of_occurrences = [(location, max(list_of__tuple_of_hashtag_and_no_of_occurrences_of_hashtag, key=itemgetter(1))) for location, list_of__tuple_of_hashtag_and_no_of_occurrences_of_hashtag in map_from_locations_to__list_of__tuple_of_hashtag_and_no_of_occurrences_of_hashtag.iteritems()]
            locations, colors = zip(*[(location, map_from_hashtag_to_color[hashtag]) for location, (hashtag, no_of_occurrences) in tuple_of_location_and__tuple_of_hashtag_and_no_of_occurrences if hashtag in map_from_hashtag_to_color])
            plt.figure()
            plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=colors, lw = 0)
            plt.title('real_data')
#            plt.show()
            
            #Plot coverge distance
            hashtag_and_coverage_distance = [(hashtag, CoverageModel.spreadDistance(points)) for hashtag, points in hashtag_and_occurrence_locations]
            location_to_hashtag_and_coverage_distance_value_map = defaultdict(list)
            for hashtag, coverage_distance in hashtag_and_coverage_distance:
                for location, coverage_distance_value in coverage_distance.iteritems(): location_to_hashtag_and_coverage_distance_value_map[location].append([hashtag, coverage_distance_value])
            location_and_colors = []
            for location, hashtag_and_coverage_distance_value in location_to_hashtag_and_coverage_distance_value_map.iteritems():
                hashtag, coverage_distance_value = max(hashtag_and_coverage_distance_value, key=itemgetter(1))
                if hashtag in map_from_hashtag_to_color: location_and_colors.append([getLocationFromLid(location.replace('_', ' ')), map_from_hashtag_to_color[hashtag] ])
            points, colors = zip(*location_and_colors)
            plt.figure()
            plotPointsOnWorldMap(points, blueMarble=False, bkcolor='#CFCFCF', c=colors, lw = 0)
            plt.title('coverage_distance')
            plt.show()


#def plotRealData():
#    for data in iterateJsonFromFile('mr_Data/1311379200'):
#        hashtag_to_points_map = dict([(h, map(lambda lid: getLocationFromLid(lid.replace('_', ' ')), zip(*occs)[1])) for h, occs in groupby(sorted(data['oc'], key=itemgetter(0)), key=itemgetter(0))])
#        hashtag_to_grouped_points = [(h, [(point, len(list(occs))) for point, occs in groupby(points, key=itemgetter(0,1))]) for h, points in hashtag_to_points_map.iteritems()]
#        locations_to_hashtag_scores_map = defaultdict(dict)
#        for hashtag, grouped_points in hashtag_to_grouped_points:
#            for location, hashtag_count in grouped_points: locations_to_hashtag_scores_map[location][hashtag] = hashtag_count
#        
#        if len(locations_to_hashtag_scores_map)>100:
#            print data['tu'], len(locations_to_hashtag_scores_map)
#            hashtag_to_color_map = getHashtagColors()
#            location_and__hashtag_and_no_of_occurrences = [(location, max(hashtags.iteritems(), key=itemgetter(1))) for location, hashtags in locations_to_hashtag_scores_map.iteritems()]
#            locations, colors = zip(*[(location, hashtag_to_color_map[hashtag]) for location, (hashtag, no_of_occurrences) in location_and__hashtag_and_no_of_occurrences if hashtag in hashtag_to_color_map])
#            plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=colors, lw = 0)
#            plt.show()

#def plotCoverageDistance():
#    for data in iterateJsonFromFile('mr_Data/1311379200'):
#        loadLocationsList()
#        hashtag_colors = getHashtagColors()
#        hashtag_and_points = [(h, map(lambda lid: getLocationFromLid(lid.replace('_', ' ')), zip(*occs)[1])) for h, occs in groupby(sorted(data['oc'], key=itemgetter(0)), key=itemgetter(0))]
#        print sorted([(hashtag, len(points)) for hashtag, points in hashtag_and_points if len(points)>25], key=itemgetter(1), reverse=True)
#        hashtag_and_coverage_distance = [(hashtag, CoverageModel.spreadDistance(points)) for hashtag, points in hashtag_and_points]
#        location_to_hashtag_and_coverage_distance_value_map = defaultdict(list)
#        for hashtag, coverage_distance in hashtag_and_coverage_distance:
#            for location, coverage_distance_value in coverage_distance.iteritems(): location_to_hashtag_and_coverage_distance_value_map[location].append([hashtag, coverage_distance_value])
#        location_and_colors = []
#        for location, hashtag_and_coverage_distance_value in location_to_hashtag_and_coverage_distance_value_map.iteritems():
#            hashtag, coverage_distance_value = max(hashtag_and_coverage_distance_value, key=itemgetter(1))
#            if hashtag in hashtag_colors: location_and_colors.append([getLocationFromLid(location.replace('_', ' ')), hashtag_colors[hashtag] ])
#        points, colors = zip(*location_and_colors)
#        plotPointsOnWorldMap(points, blueMarble=False, bkcolor='#CFCFCF', c=colors, lw = 0)
#        plt.show()
        
plotAllData()
#getHashtagColors()
#plotRealData()
#plotCoverageDistance()
