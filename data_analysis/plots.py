'''
Created on May 8, 2012

@author: krishnakamath
'''

from datetime import datetime
from settings import f_tuo_normalized_occurrence_count_and_distribution_value,\
    fld_sky_drive_data_analysis_images, f_tuo_lid_and_distribution_value,\
    f_tuo_rank_and_average_percentage_of_occurrences, \
    f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus
from library.file_io import FileIO
from library.classes import GeneralMethods
from operator import itemgetter
import matplotlib.pyplot as plt
from library.plotting import savefig
import shapefile, os
from library.geo import point_inside_polygon, getLocationFromLid
from collections import defaultdict
from library.stats import entropy, focus
import numpy as np

def iterateJsonFromFile(file):
    for data in FileIO.iterateJsonFromFile(file):
        if 'PARAMS_DICT' not in data: yield data

class CountryBoundaries:
    mf_country_to_bounding_box = {}
    @staticmethod
    def load():
        sf = shapefile.Reader(os.path.expanduser('~/SkyDrive/external_apps/TM_WORLD_BORDERS_SIMPL-0.3/TM_WORLD_BORDERS_SIMPL-0.3.shp'))
        for shape_rec in sf.shapeRecords():
            CountryBoundaries.mf_country_to_bounding_box[shape_rec.record[4]] = [[point[1], point[0]]for point in shape_rec.shape.points]
    @staticmethod
    def get_country(point):
        for country, bounding_box in \
                CountryBoundaries.mf_country_to_bounding_box.iteritems():
            if point_inside_polygon(point[0], point[1], bounding_box):
                    return country 

class DataAnalysis():
    @staticmethod
    def hashtag_distribution(input_files_start_time, input_files_end_time):
        input_file = f_tuo_normalized_occurrence_count_and_distribution_value%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'))
        output_file = fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d')) + GeneralMethods.get_method_id() + '.png'
        ltuo_s_normalized_occurrence_count_and_distribution_value \
            = sorted(
                     [(normalized_occurrence_count, distribution_value) for normalized_occurrence_count, distribution_value in iterateJsonFromFile(input_file)],
                     key=itemgetter(0), reverse=True
                     )
        x_normalized_occurrence_count, y_distribution_value = zip(*ltuo_s_normalized_occurrence_count_and_distribution_value)
        total_hashtags = float(sum(y_distribution_value))
        temp_y_distribution_value = []
        current_val = 0.0
        for distribution_value in y_distribution_value:
            current_val+=distribution_value
            temp_y_distribution_value.append(current_val/total_hashtags)
        y_distribution_value = temp_y_distribution_value
        plt.scatter(x_normalized_occurrence_count, y_distribution_value)
        plt.loglog([x_normalized_occurrence_count[0]], [y_distribution_value[0]])
        savefig(output_file)
    @staticmethod
    def occurrence_distribution_by_country(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        input_file = f_tuo_lid_and_distribution_value%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        output_file = fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) + GeneralMethods.get_method_id() + '.txt'
        GeneralMethods.runCommand('rm -rf %s'%output_file)
        CountryBoundaries.load()
        mf_country_to_occurrence_count = defaultdict(float)
        for location_count, (lid, distribution_value) in enumerate(iterateJsonFromFile(input_file)):
            print location_count
            country = CountryBoundaries.get_country(getLocationFromLid(lid.replace('_', ' ')))
            if country: mf_country_to_occurrence_count[country]+=distribution_value
        ltuo_country_and_s_occurrence_count = sorted(mf_country_to_occurrence_count.items(), key=itemgetter(1), reverse=True)
        total_occurrences = sum(zip(*ltuo_country_and_s_occurrence_count)[1])
        for country, occurrence_count in\
                 ltuo_country_and_s_occurrence_count:
            FileIO.writeToFileAsJson([country, occurrence_count, occurrence_count/float(total_occurrences)], output_file)
#    @staticmethod
#    def fraction_of_occurrences_vs_rank_of_country(input_files_start_time, input_files_end_time):
#        input_file = fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d')) + 'DataAnalysis/occurrence_distribution_by_country.txt'
#        output_file = fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d')) + GeneralMethods.get_method_id() + '.png'
#        y_fraction_of_occurrences = [fraction_of_occurrences for (_,_,fraction_of_occurrences) in
#                                        FileIO.iterateJsonFromFile(input_file)]
#        plt.semilogy(range(1,len(y_fraction_of_occurrences)+1), y_fraction_of_occurrences, lw=0, marker='o')   
#        savefig(output_file);
    @staticmethod
    def fraction_of_occurrences_vs_rank_of_location(input_files_start_time, input_files_end_time, no_of_hashtags):
        input_file = f_tuo_lid_and_distribution_value%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
        output_file = fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags) + GeneralMethods.get_method_id() + '.png'
        GeneralMethods.runCommand('rm -rf %s'%output_file)
        ltuo_lid_and_occurrene_count = []
        for lid_count, (lid, distribution_value) in enumerate(iterateJsonFromFile(input_file)):
            print lid_count
            ltuo_lid_and_occurrene_count.append([lid, distribution_value])
        total_occurrences = sum(zip(*ltuo_lid_and_occurrene_count)[1]) + 0.0
        ltuo_lid_and_r_occurrence_count = sorted(ltuo_lid_and_occurrene_count, key=itemgetter(1), reverse=True)
        y_fraction_of_occurrences = [r_occurrence_count/total_occurrences for _, r_occurrence_count in ltuo_lid_and_r_occurrence_count]
        plt.semilogy(range(1,len(y_fraction_of_occurrences)+1), y_fraction_of_occurrences, lw=0, marker='o')   
        savefig(output_file);
#    @staticmethod
#    def cumulative_fraction_of_occurrences_vs_rank_of_country(input_files_start_time, input_files_end_time):
#        input_file = fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d')) + 'DataAnalysis/occurrence_distribution_by_country.txt'
#        output_file = fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d')) + GeneralMethods.get_method_id() + '.png'
#        fraction_of_occurrences = [fraction_of_occurrences for (_,_,fraction_of_occurrences) in
#                                        FileIO.iterateJsonFromFile(input_file)]
#        y_fraction_of_occurrences = []
#        current_val = 0.0
#        for val in fraction_of_occurrences:
#            current_val+=val
#            y_fraction_of_occurrences.append(current_val)
#        plt.plot(range(1,len(y_fraction_of_occurrences)+1), y_fraction_of_occurrences, lw=0, marker='o')   
#        savefig(output_file);
    @staticmethod
    def cumulative_fraction_of_occurrences_vs_rank_of_location(input_files_start_time, input_files_end_time, no_of_hashtags):
        input_file = f_tuo_lid_and_distribution_value%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
        next_input_file = f_tuo_rank_and_average_percentage_of_occurrences%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
        output_file = fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags) + GeneralMethods.get_method_id() + '.png'
        GeneralMethods.runCommand('rm -rf %s'%output_file)
        ltuo_lid_and_occurrene_count = []
        for lid_count, (lid, distribution_value) in enumerate(iterateJsonFromFile(input_file)):
            print lid_count
            ltuo_lid_and_occurrene_count.append([lid, distribution_value])
        total_occurrences = sum(zip(*ltuo_lid_and_occurrene_count)[1]) + 0.0
        ltuo_lid_and_r_occurrence_count = sorted(ltuo_lid_and_occurrene_count, key=itemgetter(1), reverse=True)
        fraction_of_occurrences = [r_occurrence_count/total_occurrences for _, r_occurrence_count in ltuo_lid_and_r_occurrence_count]
        y_fraction_of_occurrences, current_val = [], 0.0
        for val in fraction_of_occurrences:
            current_val+=val
            y_fraction_of_occurrences.append(current_val)
        y_fraction_of_occurrences = y_fraction_of_occurrences[:50]
        
        ltuo_rank_and_average_percentage_of_occurrences = [tuo_rank_and_average_percentage_of_occurrences 
                                                          for tuo_rank_and_average_percentage_of_occurrences in iterateJsonFromFile(next_input_file)]
        ltuo_s_rank_and_average_percentage_of_occurrences = sorted(ltuo_rank_and_average_percentage_of_occurrences, key=itemgetter(0))
        y_average_percentage_of_occurrences = zip(*ltuo_s_rank_and_average_percentage_of_occurrences)[1][:50]
        
        
        x_percentage_of_locations = [x for x in range(1,len(y_fraction_of_occurrences)+1)]
        print len(x_percentage_of_locations), len(y_average_percentage_of_occurrences)
        plt.plot(x_percentage_of_locations, y_fraction_of_occurrences, lw=0, marker='o')  
        plt.plot(x_percentage_of_locations, y_average_percentage_of_occurrences, lw=0, marker='o')  
        savefig(output_file);
    @staticmethod
    def write_entropy_and_focus(input_files_start_time, input_files_end_time, no_of_hashtags):
        '''
        datetime(2011, 2, 1), datetime(2011, 2, 27), 0: 
            Global entropy:  9.0
            Global focus:  ('-23.2000_-46.4000', 0.043156708042033268)
        datetime(2011, 2, 1), datetime(2012, 4, 30), 50
            Global entropy:  9.0
            Global focus:  ('-23.2000_-46.4000', 0.033948282514978313)
        '''
        input_file = f_tuo_lid_and_distribution_value%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
        ltuo_lid_and_occurrene_count = []
        for lid_count, (lid, distribution_value) in enumerate(iterateJsonFromFile(input_file)):
            print lid_count
            ltuo_lid_and_occurrene_count.append([lid, distribution_value])
        print 'Global entropy: ', entropy(dict(ltuo_lid_and_occurrene_count))
        print 'Global focus: ', focus(dict(ltuo_lid_and_occurrene_count))
    @staticmethod
    def write_top_locations(input_files_start_time, input_files_end_time, no_of_hashtags):
        '''
        datetime(2011, 2, 1), datetime(2012, 4, 30), 50
            [['-23.2000_-46.4000', 'Sao, Paulo', 7357670.0], ['50.7500_0.0000', 'London', 6548390.0], 
                ['-5.8000_105.8500', 'Jakarata', 4536084.0], ['33.3500_-117.4500', 'Los Angeles', 3940885.0], 
                ['40.6000_-73.9500', 'New York', 3747348.0]]
        '''
        input_file = f_tuo_lid_and_distribution_value%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
        ltuo_lid_and_occurrene_count = []
        for lid_count, (lid, distribution_value) in enumerate(iterateJsonFromFile(input_file)):
            print lid_count
            ltuo_lid_and_occurrene_count.append([lid, distribution_value])
        print sorted(ltuo_lid_and_occurrene_count, key=itemgetter(1), reverse=True)[:5]
        
    @staticmethod
    def locality_measures_vs_nuber_of_occurreneces(input_files_start_time, input_files_end_time, no_of_hashtags):
        ACCURACY_NO_OF_OCCURRANCES = 25
        input_file = f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
        ltuo_hashtag_and_occurrence_count_and_entropy_and_focus = [data for data in iterateJsonFromFile(input_file)]
#        for tuo_hashtag_and_occurrence_count_and_entropy_and_focus in \
#                ltuo_hashtag_and_occurrence_count_and_entropy_and_focus:
#            print tuo_hashtag_and_occurrence_count_and_entropy_and_focus
        ltuo_entropy_and_occurrences_count = [(data[2], data[1]) for data in ltuo_hashtag_and_occurrence_count_and_entropy_and_focus]
        ltuo_focus_and_occurrences_count = [(data[3][1], data[1]) for data in ltuo_hashtag_and_occurrence_count_and_entropy_and_focus]
        def plot_graph(ltuo_locality_measure_and_occurrences_count):
            mf_normalized_occurrences_count_to_locality_measures = defaultdict(list)
            for locality_measure, occurrences_count in \
                    ltuo_locality_measure_and_occurrences_count:
                normalized_occurrence_count = int(occurrences_count/ACCURACY_NO_OF_OCCURRANCES)*ACCURACY_NO_OF_OCCURRANCES+ACCURACY_NO_OF_OCCURRANCES
                mf_normalized_occurrences_count_to_locality_measures[normalized_occurrence_count].append(locality_measure)
            x_occurrance_counts, y_locality_measures = [], []
            for k in sorted(mf_normalized_occurrences_count_to_locality_measures):
                if len(mf_normalized_occurrences_count_to_locality_measures[k]) > 10:
                    x_occurrance_counts.append(k), y_locality_measures.append(np.mean(mf_normalized_occurrences_count_to_locality_measures[k]))
            plt.scatter(x_occurrance_counts, y_locality_measures)
            plt.show()
#            y_locality_measure, x_occurrences_count = zip(*ltuo_locality_measure_and_occurrences_count)
#            plt.scatter(x_occurrences_count, y_locality_measure)
#            plt.semilogx(x_occurrences_count[0], y_locality_measure[0])
#            plt.show()
        plot_graph(ltuo_entropy_and_occurrences_count)
    @staticmethod
    def cdf_of_locality_measure(input_files_start_time, input_files_end_time, no_of_hashtags):
        def plot_graph(locality_measures):
            print locality_measures
        input_file = f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
        ltuo_hashtag_and_occurrence_count_and_entropy_and_focus = [data for data in iterateJsonFromFile(input_file)]
        entropies = zip(*ltuo_hashtag_and_occurrence_count_and_entropy_and_focus)[2]
        plot_graph(entropies)
    @staticmethod
    def run():
#        input_files_start_time, input_files_end_time, min_no_of_hashtags = datetime(2011, 2, 1), datetime(2011, 2, 27), 0
        input_files_start_time, input_files_end_time, min_no_of_hashtags = datetime(2011, 2, 1), datetime(2012, 4, 30), 50
        
#        DataAnalysis.hashtag_distribution(input_files_start_time, input_files_end_time)
#        DataAnalysis.occurrence_distribution_by_country(input_files_start_time, input_files_end_time)
#        DataAnalysis.fraction_of_occurrences_vs_rank_of_country(input_files_start_time, input_files_end_time)
        
#        DataAnalysis.occurrence_distribution_by_country(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.fraction_of_occurrences_vs_rank_of_location(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.cumulative_fraction_of_occurrences_vs_rank_of_location(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.write_entropy_and_focus(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.write_top_locations(input_files_start_time, input_files_end_time, min_no_of_hashtags)

        DataAnalysis.cdf_of_locality_measure(input_files_start_time, input_files_end_time, min_no_of_hashtags)

#        DataAnalysis.cumulative_fraction_of_occurrences_vs_rank_of_country(input_files_start_time, input_files_end_time)
        
if __name__ == '__main__':
    DataAnalysis.run()