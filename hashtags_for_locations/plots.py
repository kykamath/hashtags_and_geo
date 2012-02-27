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

for data in iterateJsonFromFile('mr_Data/timeUnitWithOccurrences'):
    hashtag_to_points_map = dict([(h, map(lambda lid: getLocationFromLid(lid.replace('_', ' ')), zip(*occs)[1])) for h, occs in groupby(sorted(data['oc'], key=itemgetter(0)), key=itemgetter(0))])
    hashtag_to_grouped_points = [(h, [(point, len(list(occs))) for point, occs in groupby(points, key=itemgetter(0,1))]) for h, points in hashtag_to_points_map.iteritems()]
    hashtagSet = hashtag_to_points_map.keys()
    locations_to_hashtag_scores_map = defaultdict(dict)
    for hashtag, grouped_points in hashtag_to_grouped_points:
        for location, hashtag_count in grouped_points: locations_to_hashtag_scores_map[location][hashtag] = hashtag_count
    
    if len(locations_to_hashtag_scores_map)>100:
        print data['tu'], len(locations_to_hashtag_scores_map)
        points, hashtags = zip(*[(location, max(hashtags.iteritems(), key=itemgetter(1))) for location, hashtags in locations_to_hashtag_scores_map.iteritems()])
        print hashtagSet
        hashtag_to_color_map = dict([(hashtag, GeneralMethods.getRandomColor()) for hashtag in hashtagSet])
        colors = [hashtag_to_color_map[hashtag] for hashtag, score in hashtags]
        plotPointsOnWorldMap(points, blueMarble=False, bkcolor='#CFCFCF', c=colors, lw = 0)
        plt.show()

