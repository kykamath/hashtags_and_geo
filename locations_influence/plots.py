'''
Created on Apr 14, 2012

@author: kykamath
'''
from models import Experiments, InfluenceMeasuringModels
import matplotlib.pyplot as plt
import numpy as np
from library.classes import GeneralMethods
from library.plotting import savefig, splineSmooth
from operator import itemgetter
from settings import analysis_folder, PARTIAL_WORLD_BOUNDARY
from library.file_io import FileIO
from library.geo import isWithinBoundingBox, getLocationFromLid,\
    plotPointsOnWorldMap
from matplotlib.patches import Ellipse

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
        InfluenceAnalysis.plot_locations_influence_on_world_map(model_ids)
if __name__ == '__main__':
    InfluenceAnalysis.run()
    
