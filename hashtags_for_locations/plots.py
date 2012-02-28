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

#for data in iterateJsonFromFile('mr_Data/timeUnitWithOccurrences'):
def plotRealData():
    for data in iterateJsonFromFile('mr_Data/1311379200'):
        hashtag_to_points_map = dict([(h, map(lambda lid: getLocationFromLid(lid.replace('_', ' ')), zip(*occs)[1])) for h, occs in groupby(sorted(data['oc'], key=itemgetter(0)), key=itemgetter(0))])
        hashtagSet = hashtag_to_points_map.keys()
#        print hashtagSet
#        exit()
        hashtag_to_grouped_points = [(h, [(point, len(list(occs))) for point, occs in groupby(points, key=itemgetter(0,1))]) for h, points in hashtag_to_points_map.iteritems()]
        locations_to_hashtag_scores_map = defaultdict(dict)
        for hashtag, grouped_points in hashtag_to_grouped_points:
            for location, hashtag_count in grouped_points: locations_to_hashtag_scores_map[location][hashtag] = hashtag_count
        
        if len(locations_to_hashtag_scores_map)>100:
            print data['tu'], len(locations_to_hashtag_scores_map)
            points, hashtags = zip(*[(location, max(hashtags.iteritems(), key=itemgetter(1))) for location, hashtags in locations_to_hashtag_scores_map.iteritems()])
            print hashtagSet
            hashtag_to_color_map = dict([(hashtag, GeneralMethods.getRandomColor()) for hashtag in hashtagSet])
            print sorted([(hashtag, len(list(occs))) for hashtag, occs in groupby(sorted(hashtags, key=itemgetter(0)), key=itemgetter(0))], key=itemgetter(1), reverse=True)
            colors = [hashtag_to_color_map[hashtag] for hashtag, score in hashtags]
            plotPointsOnWorldMap(points, blueMarble=False, bkcolor='#CFCFCF', c=colors, lw = 0)
            plt.show()

def getHashtagColors():
    for data in iterateJsonFromFile('mr_Data/1311379200'):
        hashtag_and_points = [(h, map(lambda lid: getLocationFromLid(lid.replace('_', ' ')), zip(*occs)[1])) for h, occs in groupby(sorted(data['oc'], key=itemgetter(0)), key=itemgetter(0))]
        print zip(*sorted([(hashtag, len(points)) for hashtag, points in hashtag_and_points if len(points)>25], key=itemgetter(1), reverse=True))[0]
        return dict([
                     ('replace1wordinamoviewithgrind', 'g'), 
                     ('palmeirascampeaomundial51', 'm'), 
                     ('11million', 'y'), 
                     ('prayfornorway', 'k'), 
                     ('happybirthdayselena', 'r'), 
                     ])

def plotCoverageDistance():
    for data in iterateJsonFromFile('mr_Data/1311379200'):
        loadLocationsList()
        hashtag_colors = getHashtagColors()
        hashtag_and_points = [(h, map(lambda lid: getLocationFromLid(lid.replace('_', ' ')), zip(*occs)[1])) for h, occs in groupby(sorted(data['oc'], key=itemgetter(0)), key=itemgetter(0))]
        print sorted([(hashtag, len(points)) for hashtag, points in hashtag_and_points if len(points)>25], key=itemgetter(1), reverse=True)
        hashtag_and_coverage_distance = [(hashtag, CoverageModel.spreadDistance(points)) for hashtag, points in hashtag_and_points]
        location_to_hashtag_and_coverage_distance_value_map = defaultdict(list)
        for hashtag, coverage_distance in hashtag_and_coverage_distance:
            for location, coverage_distance_value in coverage_distance.iteritems(): location_to_hashtag_and_coverage_distance_value_map[location].append([hashtag, coverage_distance_value])
        location_and_colors = []
        for location, hashtag_and_coverage_distance_value in location_to_hashtag_and_coverage_distance_value_map.iteritems():
            hashtag, coverage_distance_value = max(hashtag_and_coverage_distance_value, key=itemgetter(1))
            if hashtag in hashtag_colors: location_and_colors.append([getLocationFromLid(location.replace('_', ' ')), hashtag_colors[hashtag] ])
        points, colors = zip(*location_and_colors)
        plotPointsOnWorldMap(points, blueMarble=False, bkcolor='#CFCFCF', c=colors, lw = 0)
        plt.show()
        
#getHashtagColors()
plotCoverageDistance()
#plotRealData()
