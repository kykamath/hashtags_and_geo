'''
Created on Mar 7, 2012

@author: kykamath
'''
from checkins.settings import lidsToDistributionInSocialNetworksMapFile,\
    FOURSQUARE_ID, BRIGHTKITE_ID, GOWALLA_ID,\
    location_objects_with_minumum_checkins_at_both_location_and_users_file,\
    checkins_graph_file
from checkins.mr_modules import BOUNDARY_ID, MINIMUM_NUMBER_OF_CHECKINS_PER_USER,\
    MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION,\
    MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION_PER_USER,\
    CHECKINS_GRAPH_EDGE_WEIGHT_METHOD_ID
from checkins.analysis import iterateJsonFromFile
from library.geo import getLocationFromLid, plotPointsOnWorldMap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from operator import itemgetter
import networkx as nx
from library.graphs import plot, clusterUsingAffinityPropagation
from library.classes import GeneralMethods
from itertools import groupby

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
    def get_stats_from_valid_locations(boundary_id, minimum_number_of_checkins_per_user, minimum_number_of_checkins_per_location, minimum_number_of_checkins_per_location_per_user):
        input_file = location_objects_with_minumum_checkins_at_both_location_and_users_file%(boundary_id, minimum_number_of_checkins_per_user, minimum_number_of_checkins_per_location, minimum_number_of_checkins_per_location_per_user)
        set_of_users, no_of_checkins, no_of_locations = set(), 0.0, 0.0
        for data in iterateJsonFromFile(input_file):
            for user, _ in data['c']: set_of_users.add(user)
            no_of_checkins+=len(data['c'])
            no_of_locations+=1
        print 'No. of users: ', len(set_of_users)
        print 'No. of checkins: ', no_of_checkins
        print 'No. of valid locations: ', no_of_locations
    @staticmethod
    def load_checkins_graph(checkins_graph_file):
        graph = nx.Graph()
        for data in iterateJsonFromFile(checkins_graph_file):
            (u, v) = data['e'].split('__')
            graph.add_edge(u , v, {'w': data['w']})
        noOfClusters, clusters = clusterUsingAffinityPropagation(graph)
#        for cluster in clusters:
#            print len(cluster), cluster
            
        nodeToClusterIdMap = dict(clusters)
        colorMap = dict([(i, GeneralMethods.getRandomColor()) for i in range(noOfClusters)])
        clusters = [(c, list(l)) for c, l in groupby(sorted(clusters, key=itemgetter(1)), key=itemgetter(1))]
        points, colors = zip(*map(lambda  l: (getLocationFromLid(l.replace('_', ' ')), colorMap[nodeToClusterIdMap[l]]), graph.nodes()))
        _, m =plotPointsOnWorldMap(points[:1], s=0, lw=0, c=colors[:1], returnBaseMapObject=True)
        for u, v, data in graph.edges(data=True):
            if nodeToClusterIdMap[u]==nodeToClusterIdMap[v]:
                color, u, v, w = colorMap[nodeToClusterIdMap[u]], getLocationFromLid(u.replace('_', ' ')), getLocationFromLid(v.replace('_', ' ')), data['w']
                m.drawgreatcircle(u[1],u[0],v[1],v[0],color='k', alpha=0.5)
#        plt.title(title)
        plt.show()
        print noOfClusters
        print graph.number_of_edges()
        print graph.number_of_nodes()
#        plot(graph, draw_edge_labels=False, node_color='#A0CBE2',width=4,with_labels=False)
#        plt.show()

    @staticmethod
    def get_cluster_checkins_graph(boundary_id, minimum_number_of_checkins_per_user, minimum_number_of_checkins_per_location, minimum_number_of_checkins_per_location_per_user, checkins_graph_edge_weight_method_id):
        output_file = checkins_graph_file%(boundary_id, minimum_number_of_checkins_per_user, minimum_number_of_checkins_per_location, minimum_number_of_checkins_per_location_per_user, checkins_graph_edge_weight_method_id)
        DataAnalysis.load_checkins_graph(output_file)
    @staticmethod
    def run(boundary_id, minimum_number_of_checkins_per_user, minimum_number_of_checkins_per_location, minimum_number_of_checkins_per_location_per_user, checkins_graph_edge_weight_method_id):
#        DataAnalysis.plot_geo_distribution_in_social_networks()
#        DataAnalysis.get_stats_from_valid_locations(boundary_id, minimum_number_of_checkins_per_user, minimum_number_of_checkins_per_location, minimum_number_of_checkins_per_location_per_user)
        DataAnalysis.get_cluster_checkins_graph(boundary_id, minimum_number_of_checkins_per_user, minimum_number_of_checkins_per_location, minimum_number_of_checkins_per_location_per_user, checkins_graph_edge_weight_method_id)

if __name__ == '__main__':
    boundary_id, minimum_number_of_checkins_per_user, minimum_number_of_checkins_per_location = BOUNDARY_ID, MINIMUM_NUMBER_OF_CHECKINS_PER_USER, MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION
    minimum_number_of_checkins_per_location_per_user = MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION_PER_USER
    checkins_graph_edge_weight_method_id = CHECKINS_GRAPH_EDGE_WEIGHT_METHOD_ID
    DataAnalysis.run(boundary_id, minimum_number_of_checkins_per_user, minimum_number_of_checkins_per_location, minimum_number_of_checkins_per_location_per_user, checkins_graph_edge_weight_method_id)
    