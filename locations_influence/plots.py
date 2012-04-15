'''
Created on Apr 14, 2012

@author: kykamath
'''
from models import Experiments, InfluenceMeasuringModels
import matplotlib.pyplot as plt
import numpy as np
from library.classes import GeneralMethods
from library.plotting import savefig
from operator import itemgetter
from settings import analysis_folder
from library.file_io import FileIO
from library.geo import isWithinBoundingBox, getLocationFromLid,\
    plotPointsOnWorldMap

class InfluenceAnalysis:
    @staticmethod
    def locations_at_top_and_bottom(model_ids, no_of_locations=5):
        for model_id in model_ids:
            output_file_format = analysis_folder+'%s/'%(GeneralMethods.get_method_id())+'%s/%s.json'
            input_locations = [ 
                                ('40.6000_-73.9500', 'new_york'), 
#                                ('33.3500_-118.1750', 'los_angeles'),
#                                ('29.7250_-97.1500', 'austin'), ('30.4500_-95.7000', 'college_station') ,('29.7250_-94.9750', 'houston'),
    #                            ('39.1500_-83.3750', 'hillsboro_oh'), ('25.3750_-79.7500', 'miami'), ('-23.2000_-46.4000', 'sao_paulo'),
    #                            ('51.4750_0.0000', 'london'), ('38.4250_-76.8500', 'washington'),
    #                            ('33.3500_-84.1000', 'atlanta'), ('42.0500_-82.6500', 'detroit')
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
    def location_influence_plots(model_ids, no_of_bins_for_influence_score=100):
        for model_id in model_ids:
            output_file_format = 'images/%s/'%(GeneralMethods.get_method_id()) + '%s_%s.png'
            input_locations = [ 
                                ('40.6000_-73.9500', 'new_york'), 
    #                            ('33.3500_-118.1750', 'los_angeles'),
    #                            ('29.7250_-97.1500', 'austin'), ('30.4500_-95.7000', 'college_station'),
    #                            ('39.1500_-83.3750', 'hillsboro_oh'), ('25.3750_-79.7500', 'miami'), ('-23.2000_-46.4000', 'sao_paulo'),
    #                            ('29.7250_-94.9750', 'houston'),('51.4750_0.0000', 'london'), ('38.4250_-76.8500', 'washington'),
    #                            ('33.3500_-84.1000', 'atlanta'), ('42.0500_-82.6500', 'detroit')
                            ] 
            tuo_location_and_tuo_neighbor_location_and_influence_score = \
                Experiments.load_tuo_location_and_tuo_neighbor_location_and_pure_influence_score(model_id)
            for input_location, label in input_locations:
                for location, tuo_neighbor_location_and_influence_score in \
                        tuo_location_and_tuo_neighbor_location_and_influence_score:
                    if input_location==location:
                        figure = plt.figure()
                        size = figure.get_size_inches()
                        figure.set_size_inches( (size[0]*2, size[1]*0.5) )
                        influence_scores = zip(*tuo_neighbor_location_and_influence_score)[1]
                        no_of_influence_scores = len(influence_scores)
                        
                        hist_influence_score, bin_edges_influence_score =  np.histogram(influence_scores, no_of_bins_for_influence_score)
                        normed_hist_influence_score = map(lambda influence_score: (influence_score+0.)/no_of_influence_scores, hist_influence_score)
                        bin_edges_influence_score = list(bin_edges_influence_score)
                        normed_hist_influence_score = list(normed_hist_influence_score)
                        bin_edges_influence_score=[bin_edges_influence_score[0]]+bin_edges_influence_score+[bin_edges_influence_score[-1]]
                        normed_hist_influence_score=[0.0]+normed_hist_influence_score+[0.0]
                        x_bin_edges_influence_score, y_normed_hist_influence_score = bin_edges_influence_score[:-1], normed_hist_influence_score
    #                    x_bin_edges_influence_score, y_normed_hist_influence_score = splineSmooth(x_bin_edges_influence_score, y_normed_hist_influence_score)
                        plt.plot(x_bin_edges_influence_score, y_normed_hist_influence_score, lw=3, color='#FF9E05')
                        plt.fill_between(x_bin_edges_influence_score, y_normed_hist_influence_score, color='#FF9E05', alpha=0.3)
    #                    plt.xlim(get_new_xlim(plt.xlim()))
                        (ticks, labels) = plt.yticks()
                        plt.yticks([ticks[-2]])
                        plt.xlim(-1,1); plt.ylim(ymin=0.0)
                        savefig(output_file_format%(label, model_id))
                        break
    @staticmethod
    def get_top_influencers(model_ids, boundary, no_of_top_locations=10):
        '''
        World
            London (Center), Washington D.C, New York (Brooklyn), London (South), Detroit
            Los Angeles, New York (Babylon), Atlanta, Sao Paulo, Miami 
        ('51.4750_0.0000', '38.4250_-76.8500', '40.6000_-73.9500', '50.7500_0.0000', '42.0500_-82.6500', 
        '33.3500_-118.1750', '40.6000_-73.2250', '33.3500_-84.1000', '-23.2000_-46.4000', '25.3750_-79.7500')
        '''
        for model_id in model_ids:
            tuo_location_and_tuo_neighbor_location_and_locations_influence_score = \
                Experiments.load_tuo_location_and_tuo_neighbor_location_and_locations_influence_score(model_id, noOfInfluencers=None)
            mf_location_to_total_influence_score, set_of_locations = {}, set()
            for location, tuo_neighbor_location_and_locations_influence_score in \
                    tuo_location_and_tuo_neighbor_location_and_locations_influence_score:
                neighbor_locations, locations_influence_scores = zip(*tuo_neighbor_location_and_locations_influence_score)
                mf_location_to_total_influence_score[location] = sum(locations_influence_scores)
                set_of_locations = set_of_locations.union(set(neighbor_locations))
            no_of_locations = len(set_of_locations)
            tuples_of_location_and_mean_influence_scores = sorted([(location, total_influence_score/no_of_locations)
                                                                     for location, total_influence_score in 
                                                                     mf_location_to_total_influence_score.iteritems()],
                                                                  key=itemgetter(1), reverse=True)[:no_of_top_locations]
            print zip(*tuples_of_location_and_mean_influence_scores)[0]
    @staticmethod
    def plot_local_influencers(model_ids):
        for model_id in model_ids:
            tuples_of_boundary_and_boundary_label = [
    #            ([[-90,-180], [90, 180]], 'World', GeneralMethods.getRandomColor()),
                ([[24.527135,-127.792969], [49.61071,-59.765625]], 'USA', GeneralMethods.getRandomColor()),
                ([[10.107706,-118.660469], [26.40009,-93.699531]], 'Mexico', GeneralMethods.getRandomColor()),
                ([[-29.565473,-58.191719], [7.327985,-30.418282]], 'Brazil', GeneralMethods.getRandomColor()),
                ([[-16.6695,88.409841], [30.115057,119.698904]], 'SE-Asia', GeneralMethods.getRandomColor()),
            ]
            tuples_of_location_and_color = []
            for boundary, boundary_label, boundary_color in tuples_of_boundary_and_boundary_label:
                map_from_location_to_total_influence_score, set_of_locations = {}, set()
                tuo_location_and_tuo_neighbor_location_and_influence_score = Experiments.load_tuo_location_and_tuo_neighbor_location_and_influence_score(model_id)
                for location, tuo_neighbor_location_and_influence_score in tuo_location_and_tuo_neighbor_location_and_influence_score:
                    if isWithinBoundingBox(getLocationFromLid(location.replace('_', ' ')), boundary):
                        set_of_locations.add(location)
                        tuo_incoming_location_and_transmission_score = filter(lambda (neighbor_location, transmission_score): transmission_score<0, tuo_neighbor_location_and_influence_score)
                        for incoming_location, transmission_score in tuo_incoming_location_and_transmission_score:
                            if incoming_location not in map_from_location_to_total_influence_score: map_from_location_to_total_influence_score[incoming_location]=0.
                            map_from_location_to_total_influence_score[incoming_location]+=abs(transmission_score)
                no_of_locations = len(set_of_locations)
                tuples_of_location_and_mean_influence_scores = sorted([(location, total_influence_score/no_of_locations)
                                                                     for location, total_influence_score in 
                                                                     map_from_location_to_total_influence_score.iteritems()],
                                                                 key=itemgetter(1), reverse=True)[:10]
                locations = zip(*tuples_of_location_and_mean_influence_scores)[0]
                for location in locations: tuples_of_location_and_color.append([getLocationFromLid(location.replace('_', ' ')), boundary_color])
            locations, colors = zip(*tuples_of_location_and_color)
            plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=colors,  lw = 0, alpha=1.)
            for _, boundary_label, boundary_color in tuples_of_boundary_and_boundary_label: plt.scatter([0], [0], label=boundary_label, c=boundary_color, lw = 0)
            plt.legend(loc=3, ncol=4, mode="expand",)
#            plt.show()
            savefig('images/%s.png'%GeneralMethods.get_method_id())
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
#        InfluenceAnalysis.get_top_influencers(model_ids, [[-90,-180], [90, 180]])
        InfluenceAnalysis.plot_local_influencers(model_ids)
if __name__ == '__main__':
    InfluenceAnalysis.run()
    