'''
Created on Nov 14, 2012

@author: krishnakamath
'''

from library.classes import GeneralMethods
from library.file_io import FileIO
from library.geo import UTMConverter
from library.geo import plotPointsOnWorldMap
from library.plotting import savefig
from operator import itemgetter
from settings import f_dense_hashtag_distribution_in_locations
from settings import f_hashtag_and_location_distribution
from settings import fld_data_analysis_results
import matplotlib.pyplot as plt

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
    def top_k_locations_on_world_map(input_files_start_time, input_files_end_time, no_of_hashtags):
        output_file = fld_data_analysis_results%GeneralMethods.get_method_id() + '.png'
        ltuo_location_and_occurrence_count = []
        for location_object in\
                FileIO.iterateJsonFromFile(f_dense_hashtag_distribution_in_locations, remove_params_dict=True):
            ltuo_location_and_occurrence_count.append([
                                                      location_object['location'],
                                                      location_object['occurrences_count']
                                                    ])
        ltuo_lid_and_r_occurrence_count = sorted(ltuo_location_and_occurrence_count, key=itemgetter(1), reverse=True)
        lids = zip(*ltuo_lid_and_r_occurrence_count)[0][:200]
        points = map(UTMConverter.getLatLongUTMIdInLatLongForm, lids)
        plotPointsOnWorldMap(points, blueMarble=False, bkcolor='#CFCFCF', c='m',  lw = 0, alpha=1.)
        savefig(output_file)
#    @staticmethod
#    def write_top_locations():
#        '''
#        datetime(2011, 2, 1), datetime(2012, 4, 30), 50
#            [['-23.2000_-46.4000', 'Sao, Paulo', 7357670.0], ['50.7500_0.0000', 'London', 6548390.0], 
#                ['-5.8000_105.8500', 'Jakarata', 4536084.0], ['33.3500_-117.4500', 'Los Angeles', 3940885.0], 
#                ['40.6000_-73.9500', 'New York', 3747348.0]]
#        [('-23.2000_-46.4000', 0.033948282514978313), ('50.7500_0.0000', 0.030214265350071261), 
#        ('-5.8000_105.8500', 0.020929487343639069), ('33.3500_-117.4500', 0.018183239712985265), 
#        ('40.6000_-73.9500', 0.017290260175563586)]
#        '''
##        input_file = f_tuo_lid_and_distribution_value%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
#        ltuo_lid_and_occurrene_count = []
#        total_distribution_value = 0.0
#        for lid_count, (lid, distribution_value) in enumerate(iterateJsonFromFile(input_file)):
#            print lid_count
#            tot_distribution_value+=distribution_value
#            ltuo_lid_and_occurrene_count.append([lid, distribution_value])
#        ltuo_lid_and_occurrene_count = [(lid, occurrene_count/total_distribution_value)for lid, occurrene_count in ltuo_lid_and_occurrene_count]
#        print sorted(ltuo_lid_and_occurrene_count, key=itemgetter(1), reverse=True)[:5]
    @staticmethod
    def run():
#        DataAnalysis.hashtag_distribution_loglog()
#        DataAnalysis.hashtag_locations_distribution_loglog()
#        DataAnalysis.fraction_of_occurrences_vs_rank_of_location()
        DataAnalysis.top_k_locations_on_world_map()

if __name__ == '__main__':
    DataAnalysis.run()