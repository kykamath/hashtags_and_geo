'''
Created on Nov 14, 2012

@author: krishnakamath
'''
from collections import defaultdict
from library.classes import GeneralMethods
from library.file_io import FileIO
from library.geo import UTMConverter
from library.geo import plotPointsOnWorldMap
from library.plotting import savefig
from library.plotting import splineSmooth
from operator import itemgetter
from settings import f_dense_hashtag_distribution_in_locations
from settings import f_hashtag_and_location_distribution
from settings import f_dense_hashtags_similarity_and_lag
from settings import fld_data_analysis_results
import matplotlib.pyplot as plt
import numpy as np

class DataAnalysis():
    @staticmethod
    def hashtag_distribution_loglog():
        ltuo_no_of_hashtags_and_count = []
        for data in FileIO.iterateJsonFromFile(f_hashtag_and_location_distribution, remove_params_dict=True):
            if data[0]=='hashtag' : ltuo_no_of_hashtags_and_count.append(data[1:])
        output_file = fld_data_analysis_results%GeneralMethods.get_method_id() + '.png'
        no_of_hashtags, counts = zip(*ltuo_no_of_hashtags_and_count)
        plt.figure(num=None, figsize=(4.3, 3))
        plt.subplots_adjust(bottom=0.2, top=0.9, left=0.17)
        ax = plt.subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.scatter(no_of_hashtags, counts, c='k')
        plt.xlabel('No. of occurrences')
        plt.ylabel('No. of hashtags')
        plt.grid(True)
        plt.xlim(xmin=1/10, )
        plt.ylim(ymin=1/10, )
#        plt.show()
        savefig(output_file)
    @staticmethod
    def hashtag_locations_distribution_loglog():
        ltuo_no_of_locations_and_count = []
        for data in FileIO.iterateJsonFromFile(f_hashtag_and_location_distribution, remove_params_dict=True):
            if data[0]=='location' : ltuo_no_of_locations_and_count.append(data[1:])
        output_file = fld_data_analysis_results%GeneralMethods.get_method_id() + '.png'
        no_of_locations, counts = zip(*ltuo_no_of_locations_and_count)
        plt.figure(num=None, figsize=(4.3,3))
        plt.subplots_adjust(bottom=0.2, top=0.9, left=0.17)
        ax = plt.subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.scatter(no_of_locations, counts, c='k')
        plt.xlabel('No. of locations')
        plt.ylabel('No. of hashtags')
        plt.xlim(xmin=1/10, )
        plt.ylim(ymin=1/10, )
        plt.grid(True)
#        plt.show()
        savefig(output_file)
    @staticmethod
    def fraction_of_occurrences_vs_rank_of_location():
        output_file = fld_data_analysis_results%GeneralMethods.get_method_id() + '.png'
        ltuo_location_and_occurrence_count = []
        for location_object in\
                FileIO.iterateJsonFromFile(f_dense_hashtag_distribution_in_locations, remove_params_dict=True):
            ltuo_location_and_occurrence_count.append([
                                                      location_object['location'],
                                                      location_object['occurrences_count']
                                                    ])
#        ltuo_location_and_occurrence_count.sort(key=itemgetter(1))
#        for location, occurrence_count in ltuo_location_and_occurrence_count:
#            print location, occurrence_count
#        exit()
        total_occurrences = sum(zip(*ltuo_location_and_occurrence_count)[1]) + 0.0
        ltuo_lid_and_r_occurrence_count = sorted(ltuo_location_and_occurrence_count, key=itemgetter(1), reverse=True)
        y_fraction_of_occurrences = [r_occurrence_count/total_occurrences for _, r_occurrence_count in ltuo_lid_and_r_occurrence_count]
#        total_locations = len(y_fraction_of_occurrences)+0.
#        x_percentage_of_locations = [x/total_locations for x in range(1,len(y_fraction_of_occurrences)+1)]
        x_percentage_of_locations = range(1,len(y_fraction_of_occurrences)+1)
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9)
        plt.semilogy(x_percentage_of_locations, y_fraction_of_occurrences, lw=0, marker='o', c='k')   
        plt.ylabel('Fraction of occurrences')
        plt.xlabel('Locations ordered by their ranks')
        plt.grid(True)
        
        a = plt.axes([.55, .5, .3, .3])
#        plt.plot(range(10))
        plt.semilogy(x_percentage_of_locations, y_fraction_of_occurrences, lw=0, marker='o', c='k')   
#        plt.title('Probability')
        plt.grid(True)
        yticks = plt.yticks()
        plt.yticks([yticks[0][-1], yticks[0][0]])
#        plt.ylim(ymin=0.000001, ymax=0.15)
#        plt.ylim(ymin=-0.01, ymax=0.04)
        plt.xlim(xmin=-4, xmax=200)
        plt.setp(a)
        
#        plt.show()
        savefig(output_file)
    @staticmethod
    def top_k_locations_on_world_map():
        output_file = fld_data_analysis_results%GeneralMethods.get_method_id() + '.png'
        ltuo_location_and_occurrence_count = []
        for location_object in\
                FileIO.iterateJsonFromFile(f_dense_hashtag_distribution_in_locations, remove_params_dict=True):
            ltuo_location_and_occurrence_count.append([
                                                      location_object['location'],
                                                      location_object['occurrences_count']
                                                    ])
        ltuo_lid_and_r_occurrence_count = sorted(ltuo_location_and_occurrence_count, key=itemgetter(1), reverse=True)
        for i, d in enumerate(ltuo_lid_and_r_occurrence_count):
            print i, d
        exit()
        lids = zip(*ltuo_lid_and_r_occurrence_count)[0][:500]
        points = map(UTMConverter.getLatLongUTMIdInLatLongForm, lids)
        plotPointsOnWorldMap(points, blueMarble=False, bkcolor='#CFCFCF', c='m',  lw = 0, alpha=1.)
        savefig(output_file)
    @staticmethod
    def _plot_affinities(type):
#        TIME_UNIT_IN_SECONDS = 60*10
        mf_distance_to_affinity_scores = defaultdict(list)
        for similarity_and_lag_object in\
                FileIO.iterateJsonFromFile(f_dense_hashtags_similarity_and_lag, remove_params_dict=True):
            distance=int(similarity_and_lag_object['haversine_distance']/100)*100+100
            mf_distance_to_affinity_scores[distance].append(similarity_and_lag_object[type])
#        ltuo_distance_and_num_samples = [(distance, len(affinity_scores)) for distance, affinity_scores in mf_distance_to_affinity_scores.iteritems()]
#        ltuo_distance_and_num_samples.sort(key=itemgetter(0))
#        for distance, num_samples in ltuo_distance_and_num_samples:
#            print distance, num_samples
#        exit()
        ltuo_distance_and_affinity_score = [(distance, np.mean(affinity_scores)) 
                                            for distance, affinity_scores in mf_distance_to_affinity_scores.iteritems()
                                                if len(affinity_scores)>0]
        x_distances, y_affinity_scores = zip(*sorted(ltuo_distance_and_affinity_score, key=itemgetter(0)))
        if type=='adoption_lag': 
            y_affinity_scores = [y/(60.*60.) for y in y_affinity_scores]
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9, wspace=0, hspace=0)
        x_distances, y_affinity_scores = splineSmooth(x_distances, y_affinity_scores)
        plt.semilogx(x_distances, y_affinity_scores, c='k', lw=2)
        plt.xlim(xmin=95, xmax=15000)
        plt.grid(True)
    @staticmethod
    def content_affinity_vs_distance():
        output_file = fld_data_analysis_results%GeneralMethods.get_method_id() + '.png'
        DataAnalysis._plot_affinities('similarity')
        plt.xlabel('Distance (miles)')
        plt.ylabel('Hashtags sharing similarity')
#        plt.show()
        savefig(output_file)
    @staticmethod
    def temporal_affinity_vs_distance():
        output_file = fld_data_analysis_results%GeneralMethods.get_method_id() + '.png'
        DataAnalysis._plot_affinities('adoption_lag')
        plt.xlabel('Distance (miles)')
        plt.ylabel('Hashtag adoption lag (hours)')
#        plt.show()
        savefig(output_file)
    @staticmethod
    def run():
#        DataAnalysis.hashtag_distribution_loglog()
#        DataAnalysis.hashtag_locations_distribution_loglog()
#        DataAnalysis.fraction_of_occurrences_vs_rank_of_location()
        DataAnalysis.top_k_locations_on_world_map()
#        DataAnalysis.content_affinity_vs_distance()
#        DataAnalysis.temporal_affinity_vs_distance()

if __name__ == '__main__':
    DataAnalysis.run()