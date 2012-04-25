'''
Created on Apr 14, 2012

@author: kykamath
'''
from models import Experiments, InfluenceMeasuringModels,\
    JACCARD_SIMILARITY
import matplotlib.pyplot as plt
import numpy as np
from library.classes import GeneralMethods
from library.plotting import savefig, splineSmooth
from operator import itemgetter
from settings import analysis_folder, PARTIAL_WORLD_BOUNDARY,\
    tuo_location_and_tuo_neighbor_location_and_mf_influence_type_and_similarity_file,\
    w_extra_hashtags_tag, wout_extra_hashtags_tag
from library.file_io import FileIO
from library.geo import isWithinBoundingBox, getLocationFromLid,\
    plotPointsOnWorldMap, getLatticeLid, getHaversineDistance,\
    plot_graph_clusters_on_world_map, getHaversineDistanceForLids
from collections import defaultdict
from library.stats import filter_outliers, getOutliersRangeUsingIRQ
from scipy.stats.stats import pearsonr
from itertools import groupby
import networkx as nx
from library.graphs import clusterUsingAffinityPropagation

class InfluenceAnalysis:
    @staticmethod
    def locations_at_top_and_bottom(model_ids, no_of_locations=5):
        for model_id in model_ids:
            output_file_format = analysis_folder+'%s/'%(GeneralMethods.get_method_id())+'%s/%s.json'
            input_locations = [ 
#                                ('40.6000_-73.9500', 'new_york'), 
                                ('30.4500_-95.7000', 'college_station'), 
                            ] 
            tuo_location_and_tuo_neighbor_location_and_influence_score = \
                Experiments.load_tuo_location_and_tuo_neighbor_location_and_pure_influence_score(model_id)
            for input_location, label in input_locations:
                for location, tuo_neighbor_location_and_influence_score in \
                        tuo_location_and_tuo_neighbor_location_and_influence_score:
                    if input_location==location:
                        output_file = output_file_format%(input_location, model_id)
                        GeneralMethods.runCommand('rm -rf %s'%output_file)
                        FileIO.createDirectoryForFile(output_file)
                        FileIO.writeToFileAsJson("Bottom:", output_file)
                        for neighbor_location_and_influence_score in tuo_neighbor_location_and_influence_score[:no_of_locations]:
                            FileIO.writeToFileAsJson(neighbor_location_and_influence_score+[''], output_file)
                        FileIO.writeToFileAsJson("Top:", output_file)
                        for neighbor_location_and_influence_score in \
                                reversed(tuo_neighbor_location_and_influence_score[-no_of_locations:]):
                            FileIO.writeToFileAsJson(neighbor_location_and_influence_score+[''], output_file)
    @staticmethod
    def _plot_scores(tuo_location_and_influence_score, marking_locations, no_of_bins_for_influence_score, smooth=True):
        figure = plt.figure()
        size = figure.get_size_inches()
        figure.set_size_inches( (size[0]*2, size[1]*0.5) )
        influence_scores = zip(*tuo_location_and_influence_score)[1]
        no_of_influence_scores = len(influence_scores)
        hist_influence_score, bin_edges_influence_score =  np.histogram(influence_scores, no_of_bins_for_influence_score)
        normed_hist_influence_score = map(lambda influence_score: (influence_score+0.)/no_of_influence_scores, hist_influence_score)
        bin_edges_influence_score = list(bin_edges_influence_score)
        normed_hist_influence_score = list(normed_hist_influence_score)
        bin_edges_influence_score=[bin_edges_influence_score[0]]+bin_edges_influence_score+[bin_edges_influence_score[-1]]
        normed_hist_influence_score=[0.0]+normed_hist_influence_score+[0.0]
        x_bin_edges_influence_score, y_normed_hist_influence_score = bin_edges_influence_score[:-1], normed_hist_influence_score
        if smooth: x_bin_edges_influence_score, y_normed_hist_influence_score = splineSmooth(x_bin_edges_influence_score, y_normed_hist_influence_score)
        plt.plot(x_bin_edges_influence_score, y_normed_hist_influence_score, lw=1, color='#FF9E05')
        plt.fill_between(x_bin_edges_influence_score, y_normed_hist_influence_score, color='#FF9E05', alpha=0.3)
        mf_neighbor_location_to_influence_score = dict(tuo_location_and_influence_score)
        for marking_location in marking_locations: 
            if marking_location in mf_neighbor_location_to_influence_score:
                print marking_location, mf_neighbor_location_to_influence_score[marking_location]
#                plt.scatter([mf_neighbor_location_to_influence_score[marking_location]], [0.0005], s=20, lw=0, color=GeneralMethods.getRandomColor(), alpha=1., label=marking_location)
                plt.scatter([mf_neighbor_location_to_influence_score[marking_location]], [0.0005], s=20, lw=0, color='m', alpha=1., label=marking_location)
            else: print marking_location
#                    plt.xlim(get_new_xlim(plt.xlim()))
#        plt.legend()
        (ticks, labels) = plt.yticks()
        plt.yticks([ticks[-2]])
    @staticmethod
    def location_influence_plots(model_ids, no_of_bins_for_influence_score=100):
        for model_id in model_ids:
            output_file_format = 'images/%s/'%(GeneralMethods.get_method_id()) + '%s_%s.png'
            tuo_input_location_and_label_and_marking_locations = [ 
#                                [ '40.6000_-73.9500', 'new_york', ['-23.2000_-46.4000', '-22.4750_-42.7750', '51.4750_0.0000', '33.3500_-118.1750', '29.7250_-97.1500','30.4500_-95.7000']],
                                ['29.7250_-97.1500', 'austin',  ['-23.2000_-46.4000', '-22.4750_-42.7750', '51.4750_0.0000', '33.3500_-118.1750', '39.1500_-83.3750','30.4500_-95.7000', '40.6000_-73.9500']], 
#                                ['30.4500_-95.7000', 'college_station', ['-23.2000_-46.4000', '-22.4750_-42.7750', '51.4750_0.0000', '33.3500_-118.1750', '29.7250_-97.1500','30.4500_-95.7000', '40.6000_-73.9500']],
                            ] 
            tuo_location_and_tuo_neighbor_location_and_influence_score = \
                Experiments.load_tuo_location_and_tuo_neighbor_location_and_pure_influence_score(model_id)
            for input_location, label, marking_locations in tuo_input_location_and_label_and_marking_locations:
                for location, tuo_neighbor_location_and_influence_score in \
                        tuo_location_and_tuo_neighbor_location_and_influence_score:
                    if input_location==location:
                        InfluenceAnalysis._plot_scores(tuo_neighbor_location_and_influence_score, marking_locations, no_of_bins_for_influence_score)
                        plt.xlim(-1,1); plt.ylim(ymin=0.0)
                        plt.show()
                        savefig(output_file_format%(label, model_id))
                        break
    @staticmethod
    def global_influence_plots(ltuo_model_id_and_hashtag_tag, no_of_bins_for_influence_score=100):
        marking_locations = [
                             '18.8500_-98.6000',
#                             '2.9000_101.5000',
                             '51.4750_0.0000', 
                             '33.3500_-118.1750', 
#                             '-23.2000_-46.4000',
                            '-22.4750_-42.7750',
                            '39.1500_-83.3750',
                             '40.6000_-73.9500', 
                             '29.7250_-97.1500', 
                             '30.4500_-95.7000'
                             ]
        for model_id, hashtag_tag in ltuo_model_id_and_hashtag_tag:
            output_file = 'images/%s/'%(GeneralMethods.get_method_id()) + '%s_%s.png'%(model_id, hashtag_tag)
            tuo_location_and_global_influence_score = Experiments.load_tuo_location_and_boundary_influence_score(model_id, hashtag_tag)
            InfluenceAnalysis._plot_scores(tuo_location_and_global_influence_score, marking_locations, no_of_bins_for_influence_score, smooth=True)
            plt.ylim(ymin=0.0)
#            plt.show()
            savefig(output_file)
    @staticmethod
    def plot_local_influencers(ltuo_model_id_and_hashtag_tag):
        tuples_of_boundary_and_boundary_label = [
                ([[24.527135,-127.792969], [49.61071,-59.765625]], 'USA', GeneralMethods.getRandomColor()),
                ([[10.107706,-118.660469], [26.40009,-93.699531]], 'Mexico', GeneralMethods.getRandomColor()),
                ([[-16.6695,88.409841], [30.115057,119.698904]], 'SE-Asia', GeneralMethods.getRandomColor()),
                ([[-29.565473,-58.191719], [7.327985,-30.418282]], 'Brazil', GeneralMethods.getRandomColor()),
            ]
        for model_id, hashtag_tag in ltuo_model_id_and_hashtag_tag:
            print model_id, hashtag_tag
            tuples_of_location_and_color = []
            for boundary, boundary_label, boundary_color in tuples_of_boundary_and_boundary_label:
                tuo_location_and_influence_scores = Experiments.load_tuo_location_and_boundary_influence_score(model_id, hashtag_tag, boundary)
                tuo_location_and_influence_scores = sorted(tuo_location_and_influence_scores, key=itemgetter(1))[:10]
                locations = zip(*tuo_location_and_influence_scores)[0]
                for location in locations: tuples_of_location_and_color.append([getLocationFromLid(location.replace('_', ' ')), boundary_color])
            locations, colors = zip(*tuples_of_location_and_color)
            plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=colors,  lw = 0, alpha=1.)
            for _, boundary_label, boundary_color in tuples_of_boundary_and_boundary_label: plt.scatter([0], [0], label=boundary_label, c=boundary_color, lw = 0)
            plt.legend(loc=3, ncol=4, mode="expand",)
#            plt.show()
            savefig('images/%s/%s_%s.png'%(GeneralMethods.get_method_id(), model_id, hashtag_tag))
    @staticmethod
    def plot_global_influencers(ltuo_model_id_and_hashtag_tag):
        tuples_of_boundary_and_boundary_label = [
                ([[-90,-180], [90, 180]], 'World', 'm'),
            ]
        for model_id, hashtag_tag in ltuo_model_id_and_hashtag_tag:
            print model_id, hashtag_tag
            tuples_of_location_and_color = []
            for boundary, boundary_label, boundary_color in tuples_of_boundary_and_boundary_label:
                tuo_location_and_influence_scores = Experiments.load_tuo_location_and_boundary_influence_score(model_id, hashtag_tag, boundary)
                tuo_location_and_influence_scores = sorted(tuo_location_and_influence_scores, key=itemgetter(1))[:10]
                locations = zip(*tuo_location_and_influence_scores)[0]
                for location in locations: tuples_of_location_and_color.append([getLocationFromLid(location.replace('_', ' ')), boundary_color])
            locations, colors = zip(*tuples_of_location_and_color)
            plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=colors,  lw = 0, alpha=1.)
            for _, boundary_label, boundary_color in tuples_of_boundary_and_boundary_label: plt.scatter([0], [0], label=boundary_label, c=boundary_color, lw = 0)
#            plt.legend(loc=3, ncol=4, mode="expand",)
#            plt.show()
            savefig('images/%s/%s_%s.png'%(GeneralMethods.get_method_id(), model_id, hashtag_tag))
    @staticmethod
    def plot_locations_influence_on_world_map(ltuo_model_id_and_hashtag_tag, noOfInfluencers=10, percentage_of_locations=0.15):
        input_locations = [
#                               ('40.6000_-73.9500', 'new_york'),
#                               ('33.3500_-118.1750', 'los_angeles'),
#                               ('29.7250_-97.1500', 'austin'),
                           ('30.4500_-95.7000', 'college_station'),
                            ('-22.4750_-42.7750', 'rio'),
                           ('51.4750_0.0000', 'london'),
                         ] 
        for model_id, hashtag_tag in ltuo_model_id_and_hashtag_tag:
            tuo_location_and_tuo_neighbor_location_and_locations_influence_score = \
                    Experiments.load_tuo_location_and_tuo_neighbor_location_and_locations_influence_score(model_id, hashtag_tag, noOfInfluencers=None, influence_type=InfluenceMeasuringModels.TYPE_INCOMING_INFLUENCE)
            for input_location, label in input_locations:
                for location, tuo_neighbor_location_and_locations_influence_score in \
                        tuo_location_and_tuo_neighbor_location_and_locations_influence_score:
                    if input_location==location:
                        input_location = getLocationFromLid(input_location.replace('_', ' '))
                        output_file = 'images/%s/%s_%s/%s.png'%(GeneralMethods.get_method_id(), model_id, hashtag_tag, label)
                        number_of_outgoing_influences = int(len(tuo_neighbor_location_and_locations_influence_score)*percentage_of_locations)
                        if number_of_outgoing_influences==0: number_of_outgoing_influences=len(tuo_neighbor_location_and_locations_influence_score)
                        locations = zip(*tuo_neighbor_location_and_locations_influence_score)[0][:number_of_outgoing_influences]
                        locations = [getLocationFromLid(location.replace('_', ' ')) for location in locations]
#                        locations = filter(lambda location: isWithinBoundingBox(location, PARTIAL_WORLD_BOUNDARY), locations)
                        if locations:
                            _, m = plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c='#FF00FF', returnBaseMapObject=True, lw = 0)
                            for location in locations: 
    #                            if isWithinBoundingBox(location, PARTIAL_WORLD_BOUNDARY): 
                                m.drawgreatcircle(location[1], location[0], input_location[1], input_location[0], color='#FAA31B', lw=1., alpha=0.5)
                            plotPointsOnWorldMap([input_location], blueMarble=False, bkcolor='#CFCFCF', c='#003CFF', s=40, lw = 0)
                            FileIO.createDirectoryForFile(output_file)
                            plt.savefig(output_file)
                            plt.clf()
                        else:
                            GeneralMethods.runCommand('rm -rf %s'%output_file)
                        break
    @staticmethod
    def plot_correlation_between_influence_similarity_and_jaccard_similarity(model_ids):
        for model_id in model_ids:
            mf_influence_type_to_mf_jaccard_similarity_to_influence_similarities = {}
            for line_count, (location, tuo_neighbor_location_and_mf_influence_type_and_similarity) in \
                    enumerate(FileIO.iterateJsonFromFile(tuo_location_and_tuo_neighbor_location_and_mf_influence_type_and_similarity_file%model_id)):
                print line_count
                for neighbor_location, mf_influence_type_to_similarity in \
                        tuo_neighbor_location_and_mf_influence_type_and_similarity:
                    jaccard_similarity = round(mf_influence_type_to_similarity[JACCARD_SIMILARITY], 1)
                    for influence_type in \
                            [InfluenceMeasuringModels.TYPE_OUTGOING_INFLUENCE, InfluenceMeasuringModels.TYPE_INCOMING_INFLUENCE]:
                        if influence_type not in mf_influence_type_to_mf_jaccard_similarity_to_influence_similarities: 
                            mf_influence_type_to_mf_jaccard_similarity_to_influence_similarities[influence_type] = defaultdict(list)
                        mf_influence_type_to_mf_jaccard_similarity_to_influence_similarities[influence_type][jaccard_similarity]\
                            .append(mf_influence_type_to_similarity[influence_type])
            subplot_id = 211
            for influence_type, mf_jaccard_similarity_to_influence_similarities in \
                    mf_influence_type_to_mf_jaccard_similarity_to_influence_similarities.iteritems():
                plt.subplot(subplot_id)
                x_jaccard_similarities, y_influence_similarities = [], []
                for jaccard_similarity, influence_similarities in \
                        sorted(mf_jaccard_similarity_to_influence_similarities.iteritems(), key=itemgetter(0)):
                    influence_similarities=filter_outliers(influence_similarities)
                    if len(influence_similarities) > 10:
                        x_jaccard_similarities.append(jaccard_similarity)
                        y_influence_similarities.append(np.mean(influence_similarities))
                rho, p_value = pearsonr(x_jaccard_similarities, y_influence_similarities)
                
                plt.scatter(x_jaccard_similarities, y_influence_similarities,  
                            c = InfluenceMeasuringModels.INFLUENCE_PROPERTIES[influence_type]['color'], 
                            lw=0, s=40)
                plt.plot(x_jaccard_similarities, y_influence_similarities, 
                            c = InfluenceMeasuringModels.INFLUENCE_PROPERTIES[influence_type]['color'],  lw=2)
                if influence_type==InfluenceMeasuringModels.TYPE_INCOMING_INFLUENCE: plt.ylabel('Influencing locations similarity', fontsize=13)
                else: plt.ylabel('Influenced locations similarity', fontsize=13)
                subplot_id+=1
            plt.xlabel('Jaccard similarity', fontsize=13)
            savefig('images/%s.png'%GeneralMethods.get_method_id())
    @staticmethod
    def plot_correlation_between_influence_similarity_and_distance(model_ids, distance_accuracy=500):
        def get_larger_lid(lid): return getLatticeLid(getLocationFromLid(lid.replace('_', ' ')), 10)
        for model_id in model_ids:
            mf_influence_type_to_tuo_distance_and_similarity = defaultdict(list)
            for line_count, (location, tuo_neighbor_location_and_mf_influence_type_and_similarity) in \
                    enumerate(FileIO.iterateJsonFromFile(tuo_location_and_tuo_neighbor_location_and_mf_influence_type_and_similarity_file%model_id)):
                print line_count
                for neighbor_location, mf_influence_type_to_similarity in \
                        tuo_neighbor_location_and_mf_influence_type_and_similarity:
                    distance = getHaversineDistance(getLocationFromLid(location.replace('_', ' ')), getLocationFromLid(neighbor_location.replace('_', ' ')))
                    distance = int(distance)/distance_accuracy*distance_accuracy + distance_accuracy
                    for influence_type, similarity in mf_influence_type_to_similarity.iteritems():
                        mf_influence_type_to_tuo_distance_and_similarity[influence_type].append([distance, similarity])
            subpot_id = 211
            for influence_type in \
                    [InfluenceMeasuringModels.TYPE_OUTGOING_INFLUENCE, InfluenceMeasuringModels.TYPE_INCOMING_INFLUENCE]:
                tuo_distance_and_similarity = mf_influence_type_to_tuo_distance_and_similarity[influence_type]
                tuo_distance_and_similarities =  [(distance, zip(*ito_tuo_distance_and_similarity)[1])
                                                    for distance, ito_tuo_distance_and_similarity in groupby(
                                                            sorted(tuo_distance_and_similarity, key=itemgetter(0)),
                                                            key=itemgetter(0)
                                                        )
                                                ]
                plt.subplot(subpot_id)
                x_distances, y_similarities = [], []
                for distance, similarities in tuo_distance_and_similarities:
#                    similarities=filter_outliers(similarities)
                    x_distances.append(distance), y_similarities.append(np.mean(similarities))
    #            x_distances, y_similarities = splineSmooth(x_distances, y_similarities)
                plt.semilogy(x_distances, y_similarities, c = InfluenceMeasuringModels.INFLUENCE_PROPERTIES[influence_type]['color'], 
                         lw=2, marker = InfluenceMeasuringModels.INFLUENCE_PROPERTIES[influence_type]['marker'])
                plt.ylabel(InfluenceMeasuringModels.INFLUENCE_PROPERTIES[influence_type]['label'], fontsize=13)
                subpot_id+=1
            plt.xlabel('Distance (Miles)', fontsize=13)
#            plt.show()
            savefig('images/%s.png'%(GeneralMethods.get_method_id()))
    @staticmethod
    def influence_clusters(model_ids, min_cluster_size=15):
        influence_type = InfluenceMeasuringModels.TYPE_INCOMING_INFLUENCE
        for model_id in model_ids:
            digraph_of_location_and_location_similarity = nx.DiGraph()
            for line_count, (location, tuo_neighbor_location_and_mf_influence_type_and_similarity) in \
                        enumerate(FileIO.iterateJsonFromFile(tuo_location_and_tuo_neighbor_location_and_mf_influence_type_and_similarity_file%model_id)):
#                print line_count
                for neighbor_location, mf_influence_type_to_similarity in tuo_neighbor_location_and_mf_influence_type_and_similarity: 
                    if isWithinBoundingBox(getLocationFromLid(location.replace('_', ' ')), PARTIAL_WORLD_BOUNDARY) and \
                            isWithinBoundingBox(getLocationFromLid(neighbor_location.replace('_', ' ')), PARTIAL_WORLD_BOUNDARY):
                        digraph_of_location_and_location_similarity.add_edge(location, neighbor_location, {'w': mf_influence_type_to_similarity[influence_type]})

            no_of_clusters, tuo_location_and_cluster_id = clusterUsingAffinityPropagation(digraph_of_location_and_location_similarity)
            tuo_cluster_id_to_locations = [ (cluster_id, zip(*ito_tuo_location_and_cluster_id)[0])
                                            for cluster_id, ito_tuo_location_and_cluster_id in 
                                            groupby(
                                                  sorted(tuo_location_and_cluster_id, key=itemgetter(1)),
                                                  key=itemgetter(1)
                                                  )
                                           ]
            mf_location_to_cluster_id = dict(tuo_location_and_cluster_id)
            mf_cluster_id_to_cluster_color = dict([(i, GeneralMethods.getRandomColor()) for i in range(no_of_clusters)])
            mf_valid_locations_to_color = {}
            for cluster_id, locations in \
                    sorted(tuo_cluster_id_to_locations, key=lambda (cluster_id, locations): len(locations))[-10:]:
#                if len(locations)>min_cluster_size:
                print cluster_id, len(locations)
                for location in locations: mf_valid_locations_to_color[location] \
                    = mf_cluster_id_to_cluster_color[mf_location_to_cluster_id[location]]
            locations, colors = zip(*mf_valid_locations_to_color.iteritems())
            locations = [getLocationFromLid(location.replace('_', ' ')) for location in locations]
            _, m = plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=colors, s=0, returnBaseMapObject=True, lw = 0)
            for u, v, data in digraph_of_location_and_location_similarity.edges(data=True):
                if u in mf_valid_locations_to_color and v in mf_valid_locations_to_color \
                        and mf_location_to_cluster_id[u]==mf_location_to_cluster_id[v]:
                    color, u, v, w = mf_cluster_id_to_cluster_color[mf_location_to_cluster_id[u]], getLocationFromLid(u.replace('_', ' ')), getLocationFromLid(v.replace('_', ' ')), data['w']
                    m.drawgreatcircle(u[1], u[0], v[1], v[0], color=color, alpha=0.6)
            plt.show()
    @staticmethod
    def sharing_probability_examples(model_ids, kNoOfLocations = 3):
        tuo_target_location_and_target_location_label_and_tuo_target_nearby_location_and_nearby_location_label = [
                ('29.7250_-97.1500', 'austin' ,[('32.6250_-96.4250', 'dallas'), ('29.0000_-97.8750', 'san_antonio'), ('29.7250_-94.9750','houston')]),
            ]
        target_location = '29.7250_-97.1500'
        for model_id in model_ids:
            tuo_location_and_tuo_neighbor_location_and_sharing_affinity_score \
                = Experiments.load_tuo_location_and_tuo_neighbor_location_and_sharing_affinity_score(model_id)
#            all_locations = zip(*tuo_location_and_tuo_neighbor_location_and_sharing_affinity_score)[0]
            for target_location, target_location_label, tuo_target_nearby_location_and_nearby_location_label in \
                    tuo_target_location_and_target_location_label_and_tuo_target_nearby_location_and_nearby_location_label:
                for location, tuo_neighbor_location_and_sharing_affinity_score in \
                        tuo_location_and_tuo_neighbor_location_and_sharing_affinity_score:
                    if location==target_location:
                        mf_neighbor_location_to_sharing_affinity_score = dict(tuo_neighbor_location_and_sharing_affinity_score)
                        print [(
                                nearby_location_label, 
                                getHaversineDistanceForLids(target_nearby_location.replace('_', ' '), location.replace('_', ' ')), 
                                '%0.2f'%mf_neighbor_location_to_sharing_affinity_score[target_nearby_location]
                                )
                               for target_nearby_location, nearby_location_label in tuo_target_nearby_location_and_nearby_location_label]
                        print [(a, '%0.2f'%b)for a,b in tuo_neighbor_location_and_sharing_affinity_score[1:kNoOfLocations+1]]
    @staticmethod
    def model_comparison_with_best_model(best_tuo_model_and_hashtag_tag, ltuo_model_id_and_hashtag_tag, no_of_locations=10):
        def count_similar_pairs(current_no_of_similar_pairs, (location1, location2)):
            if location1==location2: current_no_of_similar_pairs+=1
            return current_no_of_similar_pairs
        best_model, best_hashtag_tag = best_tuo_model_and_hashtag_tag
        best_locations = Experiments.get_locations_sorted_by_boundary_influence_score(best_model, best_hashtag_tag, no_of_locations)
        for model_id, hashtag_tag in ltuo_model_id_and_hashtag_tag:
            locations = Experiments.get_locations_sorted_by_boundary_influence_score(model_id, hashtag_tag, no_of_locations)
#            metric_count = reduce(count_similar_pairs, zip(best_locations, locations), 0.0)
            metric_count = len(set(best_locations).intersection(set(locations)))
            print '%s_%s'%(model_id, hashtag_tag), metric_count/float(no_of_locations)
    @staticmethod
    def compare_With_test_set(ltuo_model_id_and_hashtag_tag):
        mf_model_id_to_misrank_accuracies = defaultdict(list)
        def to_locations_based_on_first_occurence(locations, location):
            if location not in locations: locations.append(location)
            return locations
        def get_misrank_accuracy((real_location_rank, locations_order_for_hashtag)):
            position = locations_order_for_hashtag.index(real_location_rank)
            def count_greater_than(current_count, (real_location_rank, predicted_location_rank)):
                if real_location_rank < predicted_location_rank: current_count+=1
                return current_count
            def count_lesser_than(current_count, (real_location_rank, predicted_location_rank)):
                if real_location_rank > predicted_location_rank: current_count+=1
                return current_count
            left_side_location_ranks = locations_order_for_hashtag[:position]
            right_side_location_ranks = locations_order_for_hashtag[position+1:]
            total_misranked_locations = reduce(count_greater_than, zip([real_location_rank]*len(left_side_location_ranks), left_side_location_ranks), 0.0) \
                                            + reduce(count_lesser_than, zip([real_location_rank]*len(right_side_location_ranks), right_side_location_ranks), 0.0)
            return total_misranked_locations/(len(locations_order_for_hashtag)-1)
        mf_model_id_to_locations = {}
        for model_id, hashtag_tag in ltuo_model_id_and_hashtag_tag:
            mf_model_id_to_locations[model_id] = Experiments.get_locations_sorted_by_boundary_influence_score(model_id, hashtag_tag)
        ltuo_hashtag_and_ltuo_location_and_occurrence_time = Experiments.load_ltuo_hashtag_and_ltuo_location_and_occurrence_time()
        for hashtag_count, (hashtag, ltuo_location_and_occurrence_time) in\
                enumerate(ltuo_hashtag_and_ltuo_location_and_occurrence_time):
            print hashtag_count
            ltuo_location_and_occurrence_time = sorted(ltuo_location_and_occurrence_time, key=itemgetter(1))
            locations = reduce(to_locations_based_on_first_occurence, zip(*ltuo_location_and_occurrence_time)[0], [])
            mf_location_to_hashtags_location_rank = dict(zip(locations, range(len(locations))))
            for model_id, locations in \
                    mf_model_id_to_locations.iteritems():
                models_location_rank = [ mf_location_to_hashtags_location_rank[location]
                                        for location in locations 
                                            if location in mf_location_to_hashtags_location_rank
                                    ]
                if len(models_location_rank)>1:
                    misrank_accuracies = map(
                          get_misrank_accuracy,
                          zip(models_location_rank, [models_location_rank]*len(models_location_rank))
                          )
                    mf_model_id_to_misrank_accuracies[model_id].append(np.mean(misrank_accuracies))
        for model_id, misrank_accuracies in \
                mf_model_id_to_misrank_accuracies.iteritems():
            print model_id, np.mean(misrank_accuracies)
    @staticmethod
    def run():
        model_ids = [
#                      InfluenceMeasuringModels.ID_FIRST_OCCURRENCE, 
#                      InfluenceMeasuringModels.ID_MEAN_OCCURRENCE, 
#                      InfluenceMeasuringModels.ID_AGGREGATE_OCCURRENCE, 
              InfluenceMeasuringModels.ID_WEIGHTED_AGGREGATE_OCCURRENCE,
          ]
        
        ltuo_model_id_and_hashtag_tag = [
#              (InfluenceMeasuringModels.ID_FIRST_OCCURRENCE, wout_extra_hashtags_tag),
#              (InfluenceMeasuringModels.ID_MEAN_OCCURRENCE, wout_extra_hashtags_tag),
#              (InfluenceMeasuringModels.ID_AGGREGATE_OCCURRENCE, wout_extra_hashtags_tag),
#              (InfluenceMeasuringModels.ID_WEIGHTED_AGGREGATE_OCCURRENCE, wout_extra_hashtags_tag),
              (InfluenceMeasuringModels.ID_FIRST_OCCURRENCE, w_extra_hashtags_tag),
              (InfluenceMeasuringModels.ID_MEAN_OCCURRENCE, w_extra_hashtags_tag),
              (InfluenceMeasuringModels.ID_AGGREGATE_OCCURRENCE, w_extra_hashtags_tag),
              (InfluenceMeasuringModels.ID_WEIGHTED_AGGREGATE_OCCURRENCE, w_extra_hashtags_tag),
          ]
        
#        best_tuo_model_and_hashtag_tag = (InfluenceMeasuringModels.ID_FIRST_OCCURRENCE, w_extra_hashtags_tag)
        best_tuo_model_and_hashtag_tag = (InfluenceMeasuringModels.ID_WEIGHTED_AGGREGATE_OCCURRENCE, w_extra_hashtags_tag)
        
        # DEFAULT USE w_extra_hashtags_tag AS hashtag_tag and ID_WEIGHTED_AGGREGATE_OCCURRENCE as model
#        InfluenceAnalysis.locations_at_top_and_bottom(model_ids)
#        InfluenceAnalysis.location_influence_plots(model_ids)
#        InfluenceAnalysis.global_influence_plots(ltuo_model_id_and_hashtag_tag)
#        InfluenceAnalysis.plot_local_influencers(ltuo_model_id_and_hashtag_tag)
#        InfluenceAnalysis.plot_global_influencers(ltuo_model_id_and_hashtag_tag)
#        InfluenceAnalysis.plot_locations_influence_on_world_map(ltuo_model_id_and_hashtag_tag)
#        InfluenceAnalysis.plot_correlation_between_influence_similarity_and_jaccard_similarity(model_ids)
#        InfluenceAnalysis.plot_correlation_between_influence_similarity_and_distance(model_ids)
#        InfluenceAnalysis.influence_clusters(model_ids)
#        InfluenceAnalysis.sharing_probability_examples(model_ids)
#        InfluenceAnalysis.model_comparison_with_best_model(best_tuo_model_and_hashtag_tag, ltuo_model_id_and_hashtag_tag, no_of_locations=100)
        InfluenceAnalysis.compare_With_test_set(ltuo_model_id_and_hashtag_tag)

class ModelComparison:
    @staticmethod
    def top_locations_per_model(ltuo_model_id_and_hashtag_tag):
        for model_id, hashtag_tag in \
                ltuo_model_id_and_hashtag_tag:
            print model_id, hashtag_tag
    @staticmethod
    def run():
        ltuo_model_id_and_hashtag_tag = [
              (InfluenceMeasuringModels.ID_FIRST_OCCURRENCE, wout_extra_hashtags_tag),
              (InfluenceMeasuringModels.ID_MEAN_OCCURRENCE, wout_extra_hashtags_tag),
              (InfluenceMeasuringModels.ID_AGGREGATE_OCCURRENCE, wout_extra_hashtags_tag),
              (InfluenceMeasuringModels.ID_WEIGHTED_AGGREGATE_OCCURRENCE, wout_extra_hashtags_tag),
              (InfluenceMeasuringModels.ID_WEIGHTED_AGGREGATE_OCCURRENCE, w_extra_hashtags_tag),
          ]
        ModelComparison.top_locations_per_model(ltuo_model_id_and_hashtag_tag)
        
if __name__ == '__main__':
    InfluenceAnalysis.run()
#    ModelComparison.run()
    
