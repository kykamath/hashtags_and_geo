'''
Created on Mar 7, 2012

@author: kykamath
'''
from checkins.settings import lidsToDistributionInSocialNetworksMapFile,\
    FOURSQUARE_ID, BRIGHTKITE_ID, GOWALLA_ID
from checkins.mr_modules import BOUNDARY_ID
from checkins.analysis import iterateJsonFromFile
from library.geo import getLocationFromLid, plotPointsOnWorldMap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from operator import itemgetter

class DataAnalysis:
    @staticmethod
    def plot_geo_distribution_in_social_networks():
        total_checkins = 0.0
        for social_network in [FOURSQUARE_ID, BRIGHTKITE_ID, GOWALLA_ID]:
            print social_network
            ax = plt.subplot(111)
            tuples_of_location_and_location_occurences_count = [(getLocationFromLid(data['key'].replace('_', ' ')), data['distribution'][social_network]) 
                                                         for i, data in enumerate(iterateJsonFromFile(lidsToDistributionInSocialNetworksMapFile%BOUNDARY_ID))\
                                                         if social_network in data['distribution'] and data['distribution'][social_network]>25]
            tuples_of_location_and_location_occurences_count = sorted(tuples_of_location_and_location_occurences_count, key=itemgetter(1))
            locations, colors = zip(*tuples_of_location_and_location_occurences_count)
            sc = plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=colors, cmap='cool', lw = 0)
            divider = make_axes_locatable(ax)
#            plt.title('Jaccard similarity with New York')
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(sc, cax=cax)
#            for k, v in tuples_of_location_and_location_occurences_count:
#                print social_network, k, v
#            print len(tuples_of_location_and_location_occurences_count)
            plt.show()
#            exit()
    @staticmethod
    def run():
        DataAnalysis.plot_geo_distribution_in_social_networks()
    
if __name__ == '__main__':
    DataAnalysis.run()