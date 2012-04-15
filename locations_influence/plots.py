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
    tuo_location_and_tuo_neighbor_location_and_mf_influence_type_and_similarity_file
from library.file_io import FileIO
from library.geo import isWithinBoundingBox, getLocationFromLid,\
    plotPointsOnWorldMap, getLatticeLid, getHaversineDistance
from collections import defaultdict
from library.stats import filter_outliers
from scipy.stats.stats import pearsonr

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
    def global_influence_plots(model_ids, no_of_bins_for_influence_score=100):
        label = 'global'
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
        for model_id in model_ids:
            output_file_format = 'images/%s/'%(GeneralMethods.get_method_id()) + '%s_%s.png'
            tuo_location_and_global_influence_score = Experiments.load_tuo_location_and_boundary_influence_score(model_id)
            InfluenceAnalysis._plot_scores(tuo_location_and_global_influence_score, marking_locations, no_of_bins_for_influence_score, smooth=True)
            plt.ylim(ymin=0.0)
#            plt.show()
            savefig(output_file_format%(label, model_id))
    @staticmethod
    def plot_local_influencers(model_ids):
        for model_id in model_ids:
            tuples_of_boundary_and_boundary_label = [
                ([[24.527135,-127.792969], [49.61071,-59.765625]], 'USA', GeneralMethods.getRandomColor()),
                ([[10.107706,-118.660469], [26.40009,-93.699531]], 'Mexico', GeneralMethods.getRandomColor()),
                ([[-16.6695,88.409841], [30.115057,119.698904]], 'SE-Asia', GeneralMethods.getRandomColor()),
                ([[-29.565473,-58.191719], [7.327985,-30.418282]], 'Brazil', GeneralMethods.getRandomColor()),
            ]
            tuples_of_location_and_color = []
            for boundary, boundary_label, boundary_color in tuples_of_boundary_and_boundary_label:
                tuo_location_and_influence_scores = Experiments.load_tuo_location_and_boundary_influence_score(model_id, boundary)
                tuo_location_and_influence_scores = sorted(tuo_location_and_influence_scores, key=itemgetter(1))[:10]
                locations = zip(*tuo_location_and_influence_scores)[0]
                for location in locations: tuples_of_location_and_color.append([getLocationFromLid(location.replace('_', ' ')), boundary_color])
            locations, colors = zip(*tuples_of_location_and_color)
            plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=colors,  lw = 0, alpha=1.)
            for _, boundary_label, boundary_color in tuples_of_boundary_and_boundary_label: plt.scatter([0], [0], label=boundary_label, c=boundary_color, lw = 0)
            plt.legend(loc=3, ncol=4, mode="expand",)
#            plt.show()
            savefig('images/%s.png'%GeneralMethods.get_method_id())
    @staticmethod
    def plot_locations_influence_on_world_map(model_ids, noOfInfluencers=10, percentage_of_locations=0.15):
        for model_id in model_ids:
            input_locations = [
#                               ('40.6000_-73.9500', 'new_york'),
#                               ('33.3500_-118.1750', 'los_angeles'),
#                               ('29.7250_-97.1500', 'austin'),
                               ('30.4500_-95.7000', 'college_station'),
                                ('-22.4750_-42.7750', 'rio'),
                               ('51.4750_0.0000', 'london'),
                                 ] 
            tuo_location_and_tuo_neighbor_location_and_locations_influence_score = \
                    Experiments.load_tuo_location_and_tuo_neighbor_location_and_locations_influence_score(model_id, noOfInfluencers=None, influence_type=InfluenceMeasuringModels.TYPE_INCOMING_INFLUENCE)
            for input_location, label in input_locations:
                for location, tuo_neighbor_location_and_locations_influence_score in \
                        tuo_location_and_tuo_neighbor_location_and_locations_influence_score:
                    if input_location==location:
                        input_location = getLocationFromLid(input_location.replace('_', ' '))
                        output_file = 'images/%s/%s.png'%(GeneralMethods.get_method_id(), label)
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
                        mf_jaccard_similarity_to_influence_similarities.iteritems():
                    influence_similarities=filter_outliers(influence_similarities)
                    if len(influence_similarities) > 500:
                        x_jaccard_similarities.append(jaccard_similarity)
                        y_influence_similarities.append(np.mean(influence_similarities))
                rho, p_value = pearsonr(x_jaccard_similarities, y_influence_similarities)
                plt.scatter(x_jaccard_similarities, y_influence_similarities,  
                            c = InfluenceMeasuringModels.INFLUENCE_PROPERTIES[influence_type]['color'], 
                            lw=0, s=40)
                if influence_type==InfluenceMeasuringModels.TYPE_INCOMING_INFLUENCE: plt.ylabel('Influencing locations similarity', fontsize=13)
                else: plt.ylabel('Influenced locations similarity', fontsize=13)
                subplot_id+=1
            plt.xlabel('Jaccard similarity', fontsize=13)
            savefig('images/%s.png'%GeneralMethods.get_method_id())
    @staticmethod
    def plot_influence_type_similarity_vs_distance(model_ids, distance_accuracy=500):
        def get_larger_lid(lid): return getLatticeLid(getLocationFromLid(lid.replace('_', ' ')), 10)
        for model_id in model_ids:
            mf_influence_type_to_tuo_distance_and_similarity = defaultdict(list)
            mf_larger_lid_pair_to_actual_lid_pair = {}
            mf_influence_type_to_mf_distance_to_mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences = {}
            for line_count, (location, tuo_neighbor_location_and_mf_influence_type_and_similarity) in \
                    enumerate(FileIO.iterateJsonFromFile(tuo_location_and_tuo_neighbor_location_and_mf_influence_type_and_similarity_file%model_id)):
                print line_count
                for neighbor_location, mf_influence_type_to_similarity in \
                        tuo_neighbor_location_and_mf_influence_type_and_similarity:
                    distance = getHaversineDistance(getLocationFromLid(location.replace('_', ' ')), getLocationFromLid(neighbor_location.replace('_', ' ')))
                    distance = int(distance)/distance_accuracy*distance_accuracy + distance_accuracy
                    for influence_type, similarity in mf_influence_type_to_similarity.iteritems():
                        mf_influence_type_to_tuo_distance_and_similarity[influence_type].append([distance, similarity])
                        if influence_type not in mf_influence_type_to_mf_distance_to_mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences:
                                mf_influence_type_to_mf_distance_to_mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences[influence_type] = defaultdict(dict)
                        if influence_type == InfluenceMeasuringModels.TYPE_INCOMING_INFLUENCE:
                            if distance==6000 and similarity > 0.25\
                                    or distance==9000 and similarity > 0.25 \
                                    or distance==7500 and similarity == 0.00:
                                larger_lid_pair = '__'.join(sorted([get_larger_lid(location), get_larger_lid(neighbor_location)]))
                                if larger_lid_pair not in mf_influence_type_to_mf_distance_to_mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences[influence_type][distance]:
                                    mf_influence_type_to_mf_distance_to_mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences[influence_type][distance][larger_lid_pair] = 0.
                                mf_influence_type_to_mf_distance_to_mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences[influence_type][distance][larger_lid_pair]+=1
                                if larger_lid_pair not in mf_larger_lid_pair_to_actual_lid_pair: 
                                    mf_larger_lid_pair_to_actual_lid_pair[larger_lid_pair] = '__'.join([location, neighbor_location])
    #                    elif influence_type == GeneralAnalysis.LOCATION_INFLUENCING_VECTOR:
    #                        if distance==6000 and similarity > 0.25\
    #                                or distance==9000 and similarity > 0.25 \
    #                                or distance==7500 and similarity > 0.00:
    #                            larger_lid_pair = '__'.join(sorted([get_larger_lid(location), get_larger_lid(neighbor_location)]))
    #                            if larger_lid_pair not in mf_influence_type_to_mf_distance_to_mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences[influence_type][distance]:
    #                                mf_influence_type_to_mf_distance_to_mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences[influence_type][distance][larger_lid_pair] = 0.
    #                            mf_influence_type_to_mf_distance_to_mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences[influence_type][distance][larger_lid_pair]+=1
    #                            if larger_lid_pair not in mf_larger_lid_pair_to_actual_lid_pair: 
    #                                mf_larger_lid_pair_to_actual_lid_pair[larger_lid_pair] = '__'.join([location, neighbor_location])
            mf_influence_type_to_mf_distance_to_mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences[GeneralAnalysis.LOCATION_INFLUENCING_VECTOR] = mf_influence_type_to_mf_distance_to_mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences[GeneralAnalysis.LOCATION_INFLUENCED_BY_VECTOR]
            mf_distance_to_color = dict([(6000, '#FF00D0'), (7500, '#00FF59'), (9000, '#00FFEE')])
            for influence_type in \
                    [GeneralAnalysis.LOCATION_INFLUENCING_VECTOR, GeneralAnalysis.LOCATION_INFLUENCED_BY_VECTOR]:
    #                mf_influence_type_to_tuo_distance_and_similarity:
                tuo_distance_and_similarity = mf_influence_type_to_tuo_distance_and_similarity[influence_type]
                tuo_distance_and_similarities =  [(distance, zip(*ito_tuo_distance_and_similarity)[1])
                                                    for distance, ito_tuo_distance_and_similarity in groupby(
                                                            sorted(tuo_distance_and_similarity, key=itemgetter(0)),
                                                            key=itemgetter(0)
                                                        )
                                                ]
                plt.subplot(111)
    #            el = Ellipse((2, -1), 0.5, 0.5)
    #            xy = (6000, 0.23)
    #            figure(figsize=(1,1))
                x_distances, y_similarities = [], []
                for distance, similarities in tuo_distance_and_similarities:
                    similarities=filter_outliers(similarities)
                    x_distances.append(distance), y_similarities.append(np.mean(similarities))
    #            x_distances, y_similarities = splineSmooth(x_distances, y_similarities)
                plt.plot(x_distances, y_similarities, c = GeneralAnalysis.INFLUENCE_PROPERTIES[influence_type]['color'], 
                         lw=2, marker = GeneralAnalysis.INFLUENCE_PROPERTIES[influence_type]['marker'])
                plt.xlabel('Distance (Miles)', fontsize=20)
                plt.ylabel(GeneralAnalysis.INFLUENCE_PROPERTIES[influence_type]['label'], fontsize=20)
                
                plt.ylim(ymin=0.0, ymax=0.4)
                mf_distance_to_similarity = dict(zip(x_distances, y_similarities))
                for distance, color in mf_distance_to_color.iteritems():
                    plt.plot([distance, distance], [0,mf_distance_to_similarity[distance]], '--', lw=3, c=color)
    
                a = plt.axes([0.39, 0.47, .49, .49])
                mf_distance_to_mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences = mf_influence_type_to_mf_distance_to_mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences[influence_type]
    #            mf_distance_to_mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences = mf_influence_type_to_mf_distance_to_mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences[GeneralAnalysis.LOCATION_INFLUENCED_BY_VECTOR]
                for distance, mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences in mf_distance_to_mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences.iteritems():
                    for larger_lid_pairs in mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences.keys()[:]:
                        if mf_larger_lid_pair_to_actual_lid_pair[larger_lid_pairs] not in mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences: break
                        mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences[mf_larger_lid_pair_to_actual_lid_pair[larger_lid_pairs]] = mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences[larger_lid_pairs]
                        del mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences[larger_lid_pairs]
                    tuo_of_larger_lid_pairs_and_larger_lid_pair_occurrences = sorted(
                                                                               mf_to_larger_lid_pairs_to_larger_lid_pair_occurrences.iteritems(),
                                                                               key=itemgetter(1), 
                                                                               reverse=True,
                                                                           )[:2]
                    so_locations, location_pairs = set(), []
                    for (larger_lid_pairs, _) in tuo_of_larger_lid_pairs_and_larger_lid_pair_occurrences:
                        location1, location2 = larger_lid_pairs.split('__')
                        so_locations.add(location1), so_locations.add(location2)
                        location1, location2 = getLocationFromLid(location1.replace('_', ' ')), getLocationFromLid(location2.replace('_', ' '))
                        location_pairs.append([location1, location2])
                    _, m = plotPointsOnWorldMap([getLocationFromLid(location.replace('_', ' ')) for location in so_locations], blueMarble=False, bkcolor='#CFCFCF', c=mf_distance_to_color[distance], returnBaseMapObject=True, lw = 0)
                    for location1, location2 in location_pairs: 
        #                if isWithinBoundingBox(location1, PARTIAL_WORLD_BOUNDARY) and isWithinBoundingBox(location2, PARTIAL_WORLD_BOUNDARY): 
                        m.drawgreatcircle(location1[1], location1[0], location2[1], location2[0], color=mf_distance_to_color[distance], lw=3., alpha=0.5)
    #            for distance, color in mf_distance_to_color.iteritems(): plt.scatter([0], [0], color=color, label=str(distance))
    #            plt.legend(loc=3, ncol=3, mode="expand")
                plt.setp(a)
                output_file = 'images/%s/%s.png'%(GeneralMethods.get_method_id(), GeneralAnalysis.INFLUENCE_PROPERTIES[influence_type]['id'])
                FileIO.createDirectoryForFile(output_file)
                plt.savefig(output_file)
                plt.clf()
    @staticmethod
    def run():
        model_ids = [
#                      InfluenceMeasuringModels.ID_FIRST_OCCURRENCE, 
#                      InfluenceMeasuringModels.ID_MEAN_OCCURRENCE, 
#                      InfluenceMeasuringModels.ID_AGGREGATE_OCCURRENCE, 
                      InfluenceMeasuringModels.ID_WEIGHTED_AGGREGATE_OCCURRENCE,
                  ]
#        InfluenceAnalysis.locations_at_top_and_bottom(model_ids)
#        InfluenceAnalysis.location_influence_plots(model_ids)
#        InfluenceAnalysis.global_influence_plots(model_ids)
#        InfluenceAnalysis.plot_local_influencers(model_ids)
#        InfluenceAnalysis.plot_locations_influence_on_world_map(model_ids)
        InfluenceAnalysis.plot_correlation_between_influence_similarity_and_jaccard_similarity(model_ids)
if __name__ == '__main__':
    InfluenceAnalysis.run()
    
