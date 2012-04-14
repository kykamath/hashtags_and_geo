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

class InfluenceAnalysis:
    @staticmethod
    def location_influence_plots(model_id, no_of_bins_for_influence_score=100):
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
                    print tuo_neighbor_location_and_influence_score
                    exit()
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
    def get_top_influencers(boundary, model_id, no_of_top_locations=10):
        '''
        World
            London (Center), Washington D.C, New York (Brooklyn), London (South), Detroit
            Los Angeles, New York (Babylon), Atlanta, Sao Paulo, Miami 
        ('51.4750_0.0000', '38.4250_-76.8500', '40.6000_-73.9500', '50.7500_0.0000', '42.0500_-82.6500', 
        '33.3500_-118.1750', '40.6000_-73.2250', '33.3500_-84.1000', '-23.2000_-46.4000', '25.3750_-79.7500')
        '''
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
    def run():
#        model_id = InfluenceMeasuringModels.ID_FIRST_OCCURRENCE
        model_id = InfluenceMeasuringModels.ID_AGGREGATE_OCCURRENCE
#        model_id = InfluenceMeasuringModels.ID_WEIGHTED_AGGREGATE_OCCURRENCE
        InfluenceAnalysis.location_influence_plots(model_id)
#        InfluenceAnalysis.get_top_influencers([[-90,-180], [90, 180]], model_id)

if __name__ == '__main__':
    InfluenceAnalysis.run()
    