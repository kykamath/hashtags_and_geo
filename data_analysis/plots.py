'''
Created on May 8, 2012

@author: krishnakamath
'''

from datetime import datetime
from settings import f_tuo_normalized_occurrence_count_and_distribution_value,\
    fld_sky_drive_data_analysis_images, f_tuo_lid_and_distribution_value,\
    f_tuo_rank_and_average_percentage_of_occurrences, \
    f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak, \
    f_tuo_iid_and_interval_stats, f_tuo_lid_and_ltuo_other_lid_and_temporal_distance, \
    f_tuo_lid_and_ltuo_other_lid_and_no_of_co_occurrences,\
    f_tuo_high_accuracy_lid_and_distribution, f_tuo_no_of_hashtags_and_count,\
    f_tuo_no_of_locations_and_count, f_tuo_iid_and_perct_change_of_occurrences,\
    f_tuo_no_of_peak_lids_and_count, f_hashtag_objects, \
    f_tuo_valid_focus_lid_pair_and_common_hashtag_affinity_score, \
    f_tuo_valid_focus_lid_pair_and_temporal_affinity_score
from library.file_io import FileIO
from library.classes import GeneralMethods
from operator import itemgetter
import matplotlib.pyplot as plt
from library.plotting import savefig, splineSmooth
import shapefile, os
from library.geo import point_inside_polygon, getLocationFromLid,\
    getHaversineDistance, plotPointsOnWorldMap
from collections import defaultdict
from library.stats import entropy, focus, getOutliersRangeUsingIRQ
import numpy as np
import matplotlib
from datetime import timedelta
from mr_analysis import TIME_UNIT_IN_SECONDS
from data_analysis.settings import f_tuo_normalized_iid_and_tuo_prct_of_occurrences_and_entropy_and_focus_and_coverage
from matplotlib import rc
from itertools import groupby
from mr_analysis import get_so_observed_focus_lids, get_ltuo_valid_iid_and_focus_lid
from itertools import combinations

#rc('font',**{'family':'sans-serif','sans-serif':['Times New Roman']})
#rc('axes',**{'labelweight':'bold'})
#rc('savefig',**{'dpi':100})

def iterateJsonFromFile(file):
    for data in FileIO.iterateJsonFromFile(file):
        if 'PARAMS_DICT' not in data: yield data
        
class CountryBoundaries:
    mf_country_to_bounding_box = {}
    f_mf_lid_to_country = os.path.expanduser('~/SkyDrive/external_apps/TM_WORLD_BORDERS_SIMPL-0.3/mf_lid_to_country.json')
    mf_lid_to_country = None
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
    @staticmethod
    def write_json_for_lattice_to_country():
        CountryBoundaries.load()
        GeneralMethods.runCommand('rm -rf %s'%CountryBoundaries.f_mf_lid_to_country)
        (input_files_start_time, input_files_end_time, no_of_hashtags) = datetime(2011, 2, 1), datetime(2012, 4, 30), 50
        input_file = f_tuo_lid_and_distribution_value%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
        lids = [(data[0]) for data in iterateJsonFromFile(input_file)]
        for lid_count, lid in enumerate(lids):
#            mf_lid_to_country[lid ] =  CountryBoundaries.get_country(getLocationFromLid(lid.replace('_', ' ')))
            print lid_count
            FileIO.writeToFileAsJson([lid, CountryBoundaries.get_country(getLocationFromLid(lid.replace('_', ' ')))], CountryBoundaries.f_mf_lid_to_country)
    @staticmethod
    def get_country_for_lid(lid):
        if CountryBoundaries.mf_lid_to_country==None:
            CountryBoundaries.mf_lid_to_country = {}
            for lid, country in FileIO.iterateJsonFromFile(CountryBoundaries.f_mf_lid_to_country):
                print 'Loading for: ', lid
                if country!=None:
                    CountryBoundaries.mf_lid_to_country[lid] =  country
        if lid in CountryBoundaries.mf_lid_to_country: 
            country = CountryBoundaries.mf_lid_to_country[lid]
            if country=='United States Minor Outlying Islands': country='United States'
            return country
    @staticmethod
    def run():
        CountryBoundaries.write_json_for_lattice_to_country()
        
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
            if country=='United States Minor Outlying Islands': country='United States'
            if country: mf_country_to_occurrence_count[country]+=distribution_value
        ltuo_country_and_s_occurrence_count = sorted(mf_country_to_occurrence_count.items(), key=itemgetter(1), reverse=True)
        total_occurrences = sum(zip(*ltuo_country_and_s_occurrence_count)[1])
        for country, occurrence_count in\
                 ltuo_country_and_s_occurrence_count:
            FileIO.writeToFileAsJson([country, occurrence_count, occurrence_count/float(total_occurrences)], output_file)
    @staticmethod
    def occurrence_distribution_by_world_map(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        input_file = f_tuo_high_accuracy_lid_and_distribution%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        output_file = fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) + GeneralMethods.get_method_id() + '.png'
        ltuo_lid_and_distribution_value = [data for data in iterateJsonFromFile(input_file)]
        lids, distribution_values = zip(*[
                                          tuo_lid_and_distribution_value for tuo_lid_and_distribution_value in ltuo_lid_and_distribution_value 
                                          if tuo_lid_and_distribution_value[1]>5000
                                          ])
        lids, distribution_values = zip(*sorted(zip(lids, distribution_values), key=itemgetter(1)))
        points = [getLocationFromLid(lid.replace('_', ' ')) for lid in lids]
        plotPointsOnWorldMap(points, blueMarble=False, bkcolor='#CFCFCF', c='k',  lw = 0, alpha=1.)
        savefig(output_file)
    @staticmethod
    def hashtag_distribution_loglog(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        input_file = f_tuo_no_of_hashtags_and_count%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), 0)
        output_file = fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) + GeneralMethods.get_method_id() + '.png'
        ltuo_no_of_hashtags_and_count = [data for data in iterateJsonFromFile(input_file)]
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
    def hashtag_locations_distribution_loglog(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        input_file = f_tuo_no_of_locations_and_count%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), 0)
        output_file = fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) + GeneralMethods.get_method_id() + '.png'
        ltuo_no_of_locations_and_count = [data for data in iterateJsonFromFile(input_file)]
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
        ltuo_lid_and_occurrene_count = []
        for lid_count, (lid, distribution_value) in enumerate(iterateJsonFromFile(input_file)):
            print lid_count
            ltuo_lid_and_occurrene_count.append([lid, distribution_value])
        total_occurrences = sum(zip(*ltuo_lid_and_occurrene_count)[1]) + 0.0
        ltuo_lid_and_r_occurrence_count = sorted(ltuo_lid_and_occurrene_count, key=itemgetter(1), reverse=True)
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
    def top_k_locations_on_world_map(input_files_start_time, input_files_end_time, no_of_hashtags):
        input_file = f_tuo_lid_and_distribution_value%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
        output_file = fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags) + GeneralMethods.get_method_id() + '.png'
        ltuo_lid_and_occurrene_count = []
        for lid_count, (lid, distribution_value) in enumerate(iterateJsonFromFile(input_file)):
#            print lid_count
            ltuo_lid_and_occurrene_count.append([lid, distribution_value])
        total_occurrences = sum(zip(*ltuo_lid_and_occurrene_count)[1]) + 0.0
        ltuo_lid_and_r_occurrence_count = sorted(ltuo_lid_and_occurrene_count, key=itemgetter(1), reverse=True)
        lids = zip(*ltuo_lid_and_r_occurrence_count)[0][:200]
        points = [getLocationFromLid(lid.replace('_', ' ')) for lid in lids]
        plotPointsOnWorldMap(points, blueMarble=False, bkcolor='#CFCFCF', c='m',  lw = 0, alpha=1.)
        savefig(output_file)
    @staticmethod
    def cumulative_fraction_of_occurrences_vs_rank_of_location(input_files_start_time, input_files_end_time, no_of_hashtags):
        input_file = f_tuo_lid_and_distribution_value%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
        next_input_file = f_tuo_rank_and_average_percentage_of_occurrences%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
        output_file = fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags) + GeneralMethods.get_method_id() + '.png'
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
        y_fraction_of_occurrences = y_fraction_of_occurrences
        
        ltuo_rank_and_average_percentage_of_occurrences = [tuo_rank_and_average_percentage_of_occurrences 
                                                          for tuo_rank_and_average_percentage_of_occurrences in iterateJsonFromFile(next_input_file)]
        ltuo_s_rank_and_average_percentage_of_occurrences = sorted(ltuo_rank_and_average_percentage_of_occurrences, key=itemgetter(0))
        y_average_percentage_of_occurrences = zip(*ltuo_s_rank_and_average_percentage_of_occurrences)[1]
        x_percentage_of_locations = [x for x in range(1,len(y_fraction_of_occurrences)+1)]
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9)
        plt.scatter(x_percentage_of_locations, y_fraction_of_occurrences, lw=0, s=50, marker='o', c='k', label='Global distribution')  
#        plt.scatter(x_percentage_of_locations, y_average_percentage_of_occurrences, lw=0, s=50, marker='>', c='#FF9E05', label='Average distribution')  
        plt.scatter(x_percentage_of_locations, y_average_percentage_of_occurrences, lw=0, s=50, marker='*', c='k', label='Average distribution')  
        plt.ylim(ymax=1.06)
        plt.xlim(xmin=-5., xmax=200)
        plt.ylabel('Cum. fraction of occurrences')
        plt.xlabel('Locations ordered by their ranks')
        plt.grid(True)
        plt.legend(loc=4)
        savefig(output_file)
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
        [('-23.2000_-46.4000', 0.033948282514978313), ('50.7500_0.0000', 0.030214265350071261), 
        ('-5.8000_105.8500', 0.020929487343639069), ('33.3500_-117.4500', 0.018183239712985265), 
        ('40.6000_-73.9500', 0.017290260175563586)]
        '''
        input_file = f_tuo_lid_and_distribution_value%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
        ltuo_lid_and_occurrene_count = []
        total_distribution_value = 0.0
        for lid_count, (lid, distribution_value) in enumerate(iterateJsonFromFile(input_file)):
            print lid_count
            tot_distribution_value+=distribution_value
            ltuo_lid_and_occurrene_count.append([lid, distribution_value])
        ltuo_lid_and_occurrene_count = [(lid, occurrene_count/total_distribution_value)for lid, occurrene_count in ltuo_lid_and_occurrene_count]
        print sorted(ltuo_lid_and_occurrene_count, key=itemgetter(1), reverse=True)[:5]
        
    @staticmethod
    def locality_measures_vs_nuber_of_occurreneces(input_files_start_time, input_files_end_time, no_of_hashtags):
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags) \
                + GeneralMethods.get_method_id() + '/%s.png'
        def plot_graph(ltuo_locality_measure_and_occurrences_count, id):
            mf_normalized_occurrences_count_to_locality_measures = defaultdict(list)
            for locality_measure, occurrences_count in \
                    ltuo_locality_measure_and_occurrences_count:
                normalized_occurrence_count = int(occurrences_count/ACCURACY_NO_OF_OCCURRANCES)*ACCURACY_NO_OF_OCCURRANCES+ACCURACY_NO_OF_OCCURRANCES
                mf_normalized_occurrences_count_to_locality_measures[normalized_occurrence_count].append(locality_measure)
            x_occurrance_counts, y_locality_measures = [], []
            for k in sorted(mf_normalized_occurrences_count_to_locality_measures):
                if len(mf_normalized_occurrences_count_to_locality_measures[k]) > 10:
                    x_occurrance_counts.append(k), y_locality_measures.append(np.mean(mf_normalized_occurrences_count_to_locality_measures[k]))
            plt.figure(num=None, figsize=(4.3,3.0))
            plt.subplots_adjust(bottom=0.2, top=0.9, left=0.15, wspace=0.)
            plt.scatter(x_occurrance_counts, y_locality_measures, lw=0, marker='o', c='k', s=50)
            plt.xlabel('No. of hashtag occurrences')
            plt.ylabel('Mean hashtag %s'%id)
            plt.grid(True)
            savefig(output_file%('locality_vs_occurrences_'+id))
        ACCURACY_NO_OF_OCCURRANCES = 25
        input_file = f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
        ltuo_hashtag_and_occurrence_count_and_entropy_and_focus = [data for data in iterateJsonFromFile(input_file)]
        ltuo_entropy_and_occurrences_count = [(data[2], data[1]) for data in ltuo_hashtag_and_occurrence_count_and_entropy_and_focus]
        ltuo_focus_and_occurrences_count = [(data[3][1], data[1]) for data in ltuo_hashtag_and_occurrence_count_and_entropy_and_focus]
        ltuo_coverage_and_occurrences_count = [(data[4], data[1]) for data in ltuo_hashtag_and_occurrence_count_and_entropy_and_focus]
        plot_graph(ltuo_entropy_and_occurrences_count, 'entropy')
        plot_graph(ltuo_focus_and_occurrences_count, 'focus')
        plot_graph(ltuo_coverage_and_occurrences_count, 'spread')
    @staticmethod
    def locality_measure_cdf(input_files_start_time, input_files_end_time, no_of_hashtags):
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags) \
                + GeneralMethods.get_method_id() + '/%s.png'
        def plot_graph(locality_measures, id):
            mf_apprx_to_count = defaultdict(float)
            for measure in locality_measures:
                mf_apprx_to_count[round(measure,3)]+=1
            total_hashtags = sum(mf_apprx_to_count.values())
            current_val = 0.0
            x_measure, y_distribution = [], []
            for apprx, count in sorted(mf_apprx_to_count.iteritems(), key=itemgetter(0)):
                current_val+=count
                x_measure.append(apprx)
                y_distribution.append(current_val/total_hashtags)
            plt.figure(num=None, figsize=(4.3,3))
            plt.subplots_adjust(bottom=0.2, top=0.9, left=0.15, wspace=0)
            plt.scatter(x_measure, y_distribution, lw=0, marker='o', c='k', s=25)
            plt.ylim(ymax=1.2)
            if id!='Coverage': plt.xlabel('%s'%id)
            else: plt.xlabel('%s (miles)'%id)
            plt.ylabel('CDF')
            plt.grid(True)
            savefig(output_file%('cdf_'+id))
        def plot_coverage(locality_measures, id):
            mf_apprx_to_count = defaultdict(float)
            for measure in locality_measures:
                mf_apprx_to_count[round(measure,3)]+=1
            total_hashtags = sum(mf_apprx_to_count.values())
            current_val = 0.0
            x_measure, y_distribution = [], []
            for apprx, count in sorted(mf_apprx_to_count.iteritems(), key=itemgetter(0)):
                current_val+=count
                x_measure.append(apprx)
                y_distribution.append(current_val/total_hashtags)
            plt.figure(num=None, figsize=(4.3,3))
            ax = plt.subplot(111)
            ax.set_xscale('log')
            plt.subplots_adjust(bottom=0.2, top=0.9, left=0.15, wspace=0)
            plt.scatter(x_measure, y_distribution, lw=0, marker='o', c='k', s=25)
            plt.ylim(ymax=1.2)
            if id!='Coverage': plt.xlabel('%s'%id)
            else: plt.xlabel('Spread (miles)')
            plt.ylabel('CDF')
            plt.xlim(xmin=1.)
            plt.grid(True)
            savefig(output_file%('cdf_'+id))
#            plt.show()
        input_file = f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
        ltuo_hashtag_and_occurrence_count_and_entropy_and_focus = [data for data in iterateJsonFromFile(input_file)]
        entropies = zip(*ltuo_hashtag_and_occurrence_count_and_entropy_and_focus)[2]
        focuses = zip(*ltuo_hashtag_and_occurrence_count_and_entropy_and_focus)[3]
        focuses = zip(*focuses)[1]
        coverages = zip(*ltuo_hashtag_and_occurrence_count_and_entropy_and_focus)[4]
        print 'Mean entropy: ', np.mean(entropies)
        print 'Mean focus: ', np.mean(focuses)
        print 'Median entropy: ', np.median(entropies)
        print 'Median focus: ', np.median(focuses)
        plot_graph(entropies, 'Entropy')
        plot_graph(focuses, 'Focus')
        plot_coverage(coverages, 'Spread')
    @staticmethod
    def ef_plot(input_files_start_time, input_files_end_time, no_of_hashtags):
        '''
        Global: yearof4, britneyvmas, timessquareball
        Local: arsadusit (thai flood relief campaign), volunteer4betterindia, onceuponatimeinnigeria
        '''
        input_file = f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags) \
                + GeneralMethods.get_method_id() + '.png'
        ltuo_hashtag_and_entropy_and_focus = [(data[0], data[2], data[3]) for data in iterateJsonFromFile(input_file)]
        mf_norm_focus_to_entropies = defaultdict(list)
        for _, entropy, (_, focus) in ltuo_hashtag_and_entropy_and_focus:
            mf_norm_focus_to_entropies[round(focus, 2)].append(entropy)
        plt.figure(num=None, figsize=(6,3))
        x_focus, y_entropy = zip(*[(norm_focus, np.mean(entropies)) for norm_focus, entropies in mf_norm_focus_to_entropies.iteritems() if len(entropies)>0])
        plt.subplots_adjust(bottom=0.2, top=0.9, wspace=0, hspace=0)
        plt.scatter(x_focus, y_entropy, s=50, lw=0, c='k')
        plt.xlim(xmin=-0.1, xmax=1.1)
        plt.ylim(ymin=-1, ymax=9)
        plt.xlabel('Mean hashtag focus')
        plt.ylabel('Mean hashtag entropy')
        plt.grid(True)
        savefig(output_file)
        ltuo_hashtag_and_r_entropy_and_focus = sorted(ltuo_hashtag_and_entropy_and_focus, key=itemgetter(1), reverse=True)
        ltuo_hashtag_and_r_entropy_and_s_focus = sorted(ltuo_hashtag_and_r_entropy_and_focus, key=itemgetter(2))
        hashtags = zip(*ltuo_hashtag_and_r_entropy_and_s_focus)[0]
        print list(hashtags[:20])
        print list(reversed(hashtags))[:20]
        
    @staticmethod
    def _get_country_specific_locality_info(input_files_start_time, input_files_end_time, no_of_hashtags, get_country=False):
        input_file = f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
        hashtag_stats = [data for data in iterateJsonFromFile(input_file)]
        mf_country_to_ltuo_hashtag_and_entropy_and_focus_and_coverage = defaultdict(list)
        for (hashtag, occurrence_count, entropy, focus, coverage, _) in\
                hashtag_stats:
            if get_country: country = CountryBoundaries.get_country_for_lid(focus[0])
            else: country = focus[0]
            if country!=None:
                mf_country_to_ltuo_hashtag_and_entropy_and_focus_and_coverage[country].append([hashtag, occurrence_count, entropy, focus, coverage])
        return mf_country_to_ltuo_hashtag_and_entropy_and_focus_and_coverage
    @staticmethod
    def locality_measures_locality_specific_correlation(input_files_start_time, input_files_end_time, no_of_hashtags, plot_country=False):
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags) \
                + GeneralMethods.get_method_id() + '/%s/%s.png'
        mf_country_to_ltuo_hashtag_and_entropy_and_focus_and_coverage = DataAnalysis._get_country_specific_locality_info(input_files_start_time, input_files_end_time, no_of_hashtags, get_country=plot_country)
        ltuo_country_and_r_locality_info = sorted(mf_country_to_ltuo_hashtag_and_entropy_and_focus_and_coverage.iteritems(),
                                                  key=lambda (k,v): len(v), reverse=True)[:100]
        for country, locality_info in \
                ltuo_country_and_r_locality_info:
            print country, len(locality_info)
            hashtags, occurrence_counts, entropies, focuses, coverages = zip(*locality_info)
            focuses = zip(*focuses)[1]
            mf_norm_focus_to_entropies = defaultdict(list)
            mf_norm_focus_to_coverages = defaultdict(list)
            for focus, entropy, coverage in zip(focuses,entropies, coverages):
                mf_norm_focus_to_entropies[round(focus, 2)].append(entropy)
                mf_norm_focus_to_coverages[round(focus, 2)].append(coverage)
            try:
#                plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
                plt.figure(num=None, figsize=(6,3))
                plt.subplots_adjust(bottom=0.2, top=0.9, wspace=0, hspace=0)
                x_focus, y_entropy = zip(*[(norm_focus, np.mean(entropies)) for norm_focus, entropies in mf_norm_focus_to_entropies.iteritems() if len(entropies)>5])
                _, z_coverage = zip(*[(norm_focus, np.mean(coverages)) for norm_focus, coverages in mf_norm_focus_to_coverages.iteritems() if len(coverages)>5])
                cm = matplotlib.cm.get_cmap('autumn')
                sc = plt.scatter(x_focus, y_entropy, c=z_coverage, cmap=cm, s=50, lw=0, vmin=0, vmax=2990)
                plt.xlim(xmin=-0.1, xmax=1.1)
                plt.ylim(ymin=-1, ymax=9)
                plt.colorbar(sc)
                plt.xlabel('Mean hashtag focus')
                plt.ylabel('Mean hashtag entropy')
                plt.grid(True)
                if plot_country: savefig(output_file%('country', country))
                else: savefig(output_file%('location', country))
                plt.clf()
            except Exception as e: 
                print e
                pass
    @staticmethod
    def locality_measures_location_specific_correlation_example_hashtags(input_files_start_time, input_files_end_time, no_of_hashtags, plot_country=False):
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags) \
                + GeneralMethods.get_method_id() + '.txt'
        GeneralMethods.runCommand('rm -rf %s'%output_file)
        mf_location_to_ltuo_hashtag_and_occurrences_and_entropy_and_focus_and_coverage = DataAnalysis._get_country_specific_locality_info(input_files_start_time, input_files_end_time, no_of_hashtags, get_country=plot_country)
        ltuo_location_and_r_locality_info = sorted(mf_location_to_ltuo_hashtag_and_occurrences_and_entropy_and_focus_and_coverage.iteritems(),
                                                    key=lambda (k,v): len(v), reverse=True)[:100]
        for location, ltuo_hashtag_and_occurrences_and_entropy_and_focus_and_coverage in \
                ltuo_location_and_r_locality_info:
            ltuo_hashtag_and_entropy_and_focus = [(data[0], data[2], data[3][1]) for data in ltuo_hashtag_and_occurrences_and_entropy_and_focus_and_coverage]
            ltuo_hashtag_and_r_entropy_and_focus = sorted(ltuo_hashtag_and_entropy_and_focus, key=itemgetter(1), reverse=True)
            ltuo_hashtag_and_r_entropy_and_s_focus = sorted(ltuo_hashtag_and_r_entropy_and_focus, key=itemgetter(2))
            hashtags = zip(*ltuo_hashtag_and_r_entropy_and_s_focus)[0]
            FileIO.writeToFileAsJson([location, hashtags[:5], hashtags[-5:]], output_file)
    @staticmethod
    def iid_vs_cumulative_distribution_and_peak_distribution(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        input_file = f_tuo_iid_and_interval_stats%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) \
                + GeneralMethods.get_method_id() + '/%s.png'
        ltuo_iid_and_interval_stats = [data for data in iterateJsonFromFile(input_file)]
        ltuo_s_iid_and_interval_stats = sorted(ltuo_iid_and_interval_stats, key=itemgetter(0))
        ltuo_s_iid_and_tuo_is_peak_and_cumulative_percentage_of_occurrences = [(data[0], (data[1][0], data[1][2])) for data in ltuo_s_iid_and_interval_stats]
        total_peaks = sum([data[1][0] for data in ltuo_s_iid_and_tuo_is_peak_and_cumulative_percentage_of_occurrences])+0.0
        x_iids = []
        y_is_peaks = []
        z_cumulative_percentage_of_occurrencess = []
        for (iid, (is_peak, cumulative_percentage_of_occurrences)) in ltuo_s_iid_and_tuo_is_peak_and_cumulative_percentage_of_occurrences[:100]: 
            print (iid, (is_peak, cumulative_percentage_of_occurrences)) 
            x_iids.append((iid+1)*TIME_UNIT_IN_SECONDS/60)
            y_is_peaks.append(is_peak/total_peaks)
            z_cumulative_percentage_of_occurrencess.append(cumulative_percentage_of_occurrences)
        plt.figure(num=None, figsize=(4.3,3))
        plt.subplots_adjust(bottom=0.2, top=0.9, wspace=0, hspace=0)
        plt.plot(x_iids, y_is_peaks, marker='o', c='k')
        plt.ylabel('Distribution of hashtags')
        plt.xlabel('Hashtag peak (minutes)')
        plt.grid(True)
        plt.xlim(xmax=600)
        savefig(output_file%'peaks')
        plt.clf()
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9, wspace=0, hspace=0)
        plt.plot(x_iids, z_cumulative_percentage_of_occurrencess, lw=0, marker='o', c='k')
#        plt.xlabel('Minutes')
        plt.ylabel('CDF of occurrences')
        plt.xlabel('Time (Minutes)')
        plt.grid(True)
        plt.xlim(xmax=600)
        savefig(output_file%'cdf_occurrences_peak')
#        plt.show()
    @staticmethod
    def peak_stats(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        input_file = f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) \
                + GeneralMethods.get_method_id() + '/%s.png'
        peaks = [data[5] for data in iterateJsonFromFile(input_file) if data[5] < 288]
        ltuo_peak_and_count = [(peak, len(list(ito_peaks)))
                            for peak, ito_peaks in groupby(sorted(peaks))
                            ]
        ltuo_s_peak_and_count = sorted(ltuo_peak_and_count, key=itemgetter(0))        
        current_count = 0.0
        total_count = len(peaks)+0.
        print total_count
        ltuo_peak_and_cdf = []
        for peak, count, in ltuo_s_peak_and_count:
            current_count+=count
            ltuo_peak_and_cdf.append([(peak+1)*TIME_UNIT_IN_SECONDS/(60.), current_count/total_count ])
        x_peaks, y_cdf = zip(*ltuo_peak_and_cdf)
        plt.figure(num=None, figsize=(4.3,3))
        ax=plt.subplot(111)
        ax.set_xscale('log')
        plt.subplots_adjust(bottom=0.2, top=0.9, left=0.15)
        plt.scatter(x_peaks, y_cdf, c='k', s=50, lw=0)
        plt.xlabel('Time (minutes)')
        plt.ylabel('CDF')
        plt.xlim(xmin=5.)
        plt.grid(True)
#        plt.show()             
        savefig(output_file%'peak_cdf')
        plt.clf()
        
#        plt.figure(num=None, figsize=(4.3,3))
        ax=plt.subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')
        x_peaks, y_counts = zip(*ltuo_s_peak_and_count)
        x_peaks = [(peak+1)*TIME_UNIT_IN_SECONDS/(60.) for peak in x_peaks]
        y_counts = [count/total_count for count in y_counts]
        plt.scatter(x_peaks, y_counts, c='k', s=50, lw=0)
        plt.xlabel('Time (minutes)')
        plt.ylabel('Distribution of hashtags')
        plt.xlim(xmin=5)
        plt.ylim(ymax=1., ymin=0.00005)
        plt.grid(True)
        savefig(output_file%'peak_dist')
    @staticmethod
    def occurrence_decay(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        input_file = f_tuo_iid_and_perct_change_of_occurrences%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) \
                + GeneralMethods.get_method_id() + '/%s.png'
        ltuo_iid_and_perct_change_of_occurrences = [data for data in iterateJsonFromFile(input_file)]
        ltuo_s_iid_and_perct_change_of_occurrences = sorted(ltuo_iid_and_perct_change_of_occurrences,
                                                          key = itemgetter(0))[:200]
        x_iids, y_perct_change_of_occurrences = zip(*ltuo_s_iid_and_perct_change_of_occurrences)
        plt.scatter(x_iids, y_perct_change_of_occurrences)
#        plt.xlim(xmax=288)
        plt.show()
    @staticmethod
    def norm_iid_vs_locality_measuers(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        input_file = f_tuo_normalized_iid_and_tuo_prct_of_occurrences_and_entropy_and_focus_and_coverage%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) \
                + GeneralMethods.get_method_id() + '/%s.png'
        ltuo_normalized_iid_and_tuo_prct_of_occurrences_and_entropy_and_focus_and_coverage = \
            [data for data in iterateJsonFromFile(input_file)]
        x_normalized_iids, y_entropies, y_focuses, y_distance_from_overall_entropy, y_distance_from_overall_focus, y_coverages = \
                                                     zip(*sorted([(data[0]*TIME_UNIT_IN_SECONDS/60, data[1][1], data[1][2], data[1][4], data[1][5], data[1][3]) 
                                                                      for data in 
                                                                        ltuo_normalized_iid_and_tuo_prct_of_occurrences_and_entropy_and_focus_and_coverage
                                                                  ])
                                                        )
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9)
        plt.subplot(111)
        plt.xlim(xmin=-20, xmax=200)
#        plt.ylim(ymin=0.5, ymax=1.0)
        plt.plot(x_normalized_iids, y_coverages,  lw=1, c='k')
        plt.scatter(x_normalized_iids, y_coverages, lw=0, marker='o', s=50, c='k')
        plt.ylabel('Interval coverage')
        plt.xlabel('Minutes since peak')
        plt.grid(True)
        savefig(output_file%'coverage')
        plt.clf() 
        
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9)
        plt.subplot(111)
        plt.xlim(xmin=-20, xmax=120)
        plt.ylim(ymin=0.65, ymax=0.95)
        plt.plot(x_normalized_iids, y_entropies,  lw=1, c='k')
        plt.scatter(x_normalized_iids, y_entropies, lw=0, marker='o', s=50, c='k')
        plt.ylabel('Interval entropy')
        plt.xlabel('Minutes since peak')
        plt.grid(True)
        savefig(output_file%'entropy')
        plt.clf() 
        
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9)
        plt.subplot(111)
        plt.xlim(xmin=-20, xmax=400)
        plt.ylim(ymin=1, ymax=3)
        plt.plot(x_normalized_iids, y_distance_from_overall_entropy, lw=1, c='k')                               
        plt.scatter(x_normalized_iids,  y_distance_from_overall_entropy, marker='o', s=50, c='k')
        plt.xlabel('Minutes since peak')
        plt.ylabel('Distance from overall entropy')
        plt.grid(True)
        savefig(output_file%'distace_from_overall_entropy')
        plt.clf()   
        
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9)
        plt.subplot(111)
        plt.xlim(xmin=-20, xmax=120)
        plt.ylim(ymin=0.738, ymax=0.79)
        plt.plot(x_normalized_iids, y_focuses, lw=1, c='k')
        plt.scatter(x_normalized_iids, y_focuses, lw=1, marker='o', s=50, c='k')     
        plt.xlabel('Minutes since peak')
        plt.ylabel('Interval focus')
        plt.grid(True)
        savefig(output_file%'focus')
        plt.clf()
        
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9)
        plt.subplot(111)
        plt.xlim(xmin=-20, xmax=400)
        plt.ylim(ymin=-0.43, ymax=-0.19)
        plt.plot(x_normalized_iids, y_distance_from_overall_focus, lw=1, c='k')                               
        plt.scatter(x_normalized_iids, y_distance_from_overall_focus, marker='o', s=50, c='k')   
        plt.xlabel('Minutes since peak')
        plt.ylabel('Distance from overall focus')
        plt.grid(True)
        savefig(output_file%'distace_from_overall_focus')

    @staticmethod
    def ef_plots_for_peak(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        '''
        Global anticipated: mtvema, happypiday, hockeyseason, endof2011
        Local anticipated: ties2012, ache2012, omah2012q8
        Global organic: chilhoodmemories, beliebersarewinning, belieberteam
        Local organic: expoparademinas, colegiocomercial, brazilloversgagaqueenofpop
        '''
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) \
                + GeneralMethods.get_method_id() + '/%s.png'
        def getNearestNumber(num): return  (int(round(num,2)*100/100)*100 + int((round(num,2)*100%100)/3)*3)/100.
        def plot_correlation_ef_plot(condition, id, hashtags, focuses, entropies, peaks):
            mf_norm_focus_to_entropies = defaultdict(list)
            mf_norm_focus_to_peaks = defaultdict(list)
    #        plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
            for focus, entropy, peak in zip(focuses,entropies, peaks):
                if condition(peak):
                    mf_norm_focus_to_entropies[round(focus, 2)].append(entropy)
                    mf_norm_focus_to_peaks[round(focus, 2)].append(peak)
            x_focus, y_entropy = zip(*[(norm_focus, np.mean(entropies)) for norm_focus, entropies in mf_norm_focus_to_entropies.iteritems() if len(entropies)>5])
            _, z_peak = zip(*[(norm_focus, np.mean(peaks)*TIME_UNIT_IN_SECONDS/60) for norm_focus, peaks in mf_norm_focus_to_peaks.iteritems() if len(peaks)>5])
            plt.figure(num=None, figsize=(6,3))
            plt.subplots_adjust(bottom=0.2, top=0.9, wspace=0, hspace=0)
            cm = matplotlib.cm.get_cmap('cool')
            sc = plt.scatter(x_focus, y_entropy, c=z_peak, cmap=cm, s=50, lw=0,)
            plt.colorbar(sc)
            plt.xlim(xmin=-0.1, xmax=1.1)
            plt.ylim(ymin=-1, ymax=9)
            plt.xlabel('Mean hashtag focus')
            plt.ylabel('Mean hashtag entropy')
            plt.grid(True)
            savefig(output_file%id)
            ltuo_hashtag_and_entropy_and_focus = zip(hashtags, entropies, focuses)
            ltuo_hashtag_and_r_entropy_and_focus = sorted(ltuo_hashtag_and_entropy_and_focus, key=itemgetter(1), reverse=True)
            ltuo_hashtag_and_r_entropy_and_s_focus = sorted(ltuo_hashtag_and_r_entropy_and_focus, key=itemgetter(2))
            hashtags = zip(*ltuo_hashtag_and_r_entropy_and_s_focus)[0]
            print id, list(hashtags)
            print id, list(reversed(hashtags))
#            plt.show()
        input_file = f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        ltuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak = [data for data in iterateJsonFromFile(input_file)]
        hashtags, _, entropies, focuses, _, peaks = zip(*ltuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak)
        focuses = zip(*focuses)[1]
        def gt_288(peak):
            if 288>peak and peak<1008: return True
        def lt_6(peak):
            if peak < 6: return True
        def lt_144(peak):
            if peak < 144: return True
        plot_correlation_ef_plot(gt_288, 'gt_288', hashtags, focuses, entropies, peaks)
        plot_correlation_ef_plot(lt_6, 'lt_6', hashtags, focuses, entropies, peaks)
    @staticmethod
    def distance_vs_no_of_common_hashtags(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        input_file = f_tuo_lid_and_ltuo_other_lid_and_no_of_co_occurrences%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) \
                + GeneralMethods.get_method_id() + '.png'
        ltuo_lid_other_lid_and_no_of_co_occurrences = [data for data in iterateJsonFromFile(input_file)]
        mf_distance_to_total_co_occurrences = defaultdict(float)
        for lid_other_lid, no_of_co_occurrences in ltuo_lid_other_lid_and_no_of_co_occurrences:
            lid1, lid2 = lid_other_lid.split(':ilab:')
            distance = getHaversineDistance(getLocationFromLid(lid1.replace('_', ' ')), getLocationFromLid(lid2.replace('_', ' ')))
            distance=int(distance/100)*100+100
            mf_distance_to_total_co_occurrences[distance]+=no_of_co_occurrences
        total_occurrences = sum(mf_distance_to_total_co_occurrences.values())
        x_distance, y_total_co_occurrences = zip(*sorted(mf_distance_to_total_co_occurrences.items(), key=itemgetter(0)))
        y_total_co_occurrences = [y/total_occurrences for y in y_total_co_occurrences]
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9, wspace=0, hspace=0)
        x_distance, y_total_co_occurrences = splineSmooth(x_distance, y_total_co_occurrences)
        plt.semilogx(x_distance, y_total_co_occurrences, c='k', lw=2)
        plt.xlim(xmin=95, xmax=15000)
        plt.grid(True)
        plt.xlabel('Distance (miles)')
        plt.ylabel('Percentage of shared hastags')
#        plt.show()
        savefig(output_file)
    @staticmethod
    def distance_vs_temporal_distance(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        input_file = f_tuo_lid_and_ltuo_other_lid_and_temporal_distance%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) \
                + GeneralMethods.get_method_id() + '.png'
        ltuo_lid_other_lid_and_temporal_distance = [data for data in iterateJsonFromFile(input_file)]
        mf_distance_to_temporal_distances = defaultdict(list)
        close_lids = []
        for lid_other_lid, temporal_distance in ltuo_lid_other_lid_and_temporal_distance:
            lid1, lid2 = lid_other_lid.split(':ilab:')
            distance = getHaversineDistance(getLocationFromLid(lid1.replace('_', ' ')), getLocationFromLid(lid2.replace('_', ' ')))
            distance=int(distance/100)*100+100
            if distance==4400: 
                close_lids.append([getLocationFromLid(lid1.replace('_', ' ')), getLocationFromLid(lid2.replace('_', ' '))])
#                print lid1, lid2
#                close_lids.append(lid1), close_lids.append(lid2)
            mf_distance_to_temporal_distances[distance].append(temporal_distance)
#        total_occurrences = sum(mf_distance_to_total_co_occurrences.values())
        x_distance, y_temporal_distances = zip(*[(x,y)
                                                 for x, y in
                                                    sorted(mf_distance_to_temporal_distances.items(), key=itemgetter(0))
                                                 if len(y)>60
                                                 ])
        y_temporal_distances = [np.mean(y) for y in y_temporal_distances]
        y_temporal_distances = [y*TIME_UNIT_IN_SECONDS/(60.*60.)for y in y_temporal_distances]
#        for x, y in zip(x_distance, y_temporal_distances): print x, y
#        y_total_co_occurrences = [y/total_occurrences for y in y_total_co_occurrences]
        x_distance, y_temporal_distances = splineSmooth(x_distance, y_temporal_distances)
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9, wspace=0, hspace=0)
        plt.semilogx(x_distance, y_temporal_distances, c='k', lw=2)
        plt.xlim(xmin=95, xmax=15000)
        plt.grid(True)
        plt.xlabel('Distance (miles)')
        plt.ylabel('Focus times diff. (hours)')
#        plt.show()
        print len(close_lids)
        savefig(output_file)
        
        
#        locations = [lid for lids in close_lids for lid in lids]
##        plotPointsOnWorldMap(points, blueMarble=False, bkcolor='#CFCFCF', c='m',  lw = 0, alpha=1.)
#        _, m = plotPointsOnWorldMap(locations, resolution= 'l', blueMarble=False, bkcolor='#ffffff', c='#FF00FF', returnBaseMapObject=True, lw = 0)
#        for location1, location2 in close_lids: 
##                            if isWithinBoundingBox(location, PARTIAL_WORLD_BOUNDARY): 
#            m.drawgreatcircle(location1[1], location1[0], location2[1], location2[0], color='#FAA31B', lw=1., alpha=0.5)
#        plt.show()
    @staticmethod
    def write_examples_of_locations_at_different_ends_of_distance_spectrum(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        '''
        ['2.9000_101.5000:ilab:24.6500_-79.7500', [79.0, 12142.435092213409]] Kuala Laumpur - Miami
        '''
        input_file = f_tuo_lid_and_ltuo_other_lid_and_no_of_co_occurrences%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) \
                + GeneralMethods.get_method_id() + '.png'
        ltuo_lid_other_lid_and_no_of_co_occurrences = [data for data in iterateJsonFromFile(input_file)]
        ltuo_lid_other_lid_and_tuo_no_of_co_occurrences_and_distance = []
        for lid_other_lid, no_of_co_occurrences in ltuo_lid_other_lid_and_no_of_co_occurrences:
            lid1, lid2 = lid_other_lid.split(':ilab:')
            distance = getHaversineDistance(getLocationFromLid(lid1.replace('_', ' ')), getLocationFromLid(lid2.replace('_', ' ')))
            ltuo_lid_other_lid_and_tuo_no_of_co_occurrences_and_distance.append(
                                                                                    [lid_other_lid, [no_of_co_occurrences, distance]]
                                                                                )
        ltuo_lid_other_lid_and_s_tuo_no_of_co_occurrences_and_distance = \
            sorted(ltuo_lid_other_lid_and_tuo_no_of_co_occurrences_and_distance, key=lambda (_, (no_of_occurrences, __)): no_of_occurrences)
        ltuo_lid_other_lid_and_s_tuo_no_of_co_occurrences_and_s_distance = \
            sorted(ltuo_lid_other_lid_and_s_tuo_no_of_co_occurrences_and_distance, key=lambda (_, (__, distance)): distance)
#        filtered_ltuo_lid_other_lid_and_s_tuo_no_of_co_occurrences_and_s_distance = \
#            [for ltuo_lid_other_lid_and_s_tuo_no_of_co_occurrences_and_distance in ]
    @staticmethod
    def peak_lids_dist(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        '''
        Median = 7.0
        '''
        input_file = f_tuo_no_of_peak_lids_and_count%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) \
                + GeneralMethods.get_method_id() + '.png'
        ltuo_no_of_peak_lids_and_count = [data for data in iterateJsonFromFile(input_file)]
        ltuo_s_no_of_peak_lids_and_count = sorted(ltuo_no_of_peak_lids_and_count, key=itemgetter(0))
        x_no_of_peak_lids, y_count = zip(*ltuo_s_no_of_peak_lids_and_count)
        total_count = sum(y_count) + 0.
        temp_y_count = []
        current_count=0.
        for count in y_count:
            current_count+=count
            temp_y_count.append(current_count/total_count)
        y_count = temp_y_count
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9, wspace=0, hspace=0)
        ax = plt.subplot(111)
        plt.scatter(x_no_of_peak_lids, y_count, s=50, c='k')
        plt.xlim(xmin=1)
        plt.ylim(ymin=0)
        ax.set_xscale('log')
        plt.xlabel('No. of focus locations')
        plt.ylabel('CDF')
        plt.grid(True)
#        plt.show()
        print np.median([x for x, y in ltuo_s_no_of_peak_lids_and_count for i in range(y)])
        savefig(output_file)

    @staticmethod
    def coverage_vs_spatial_properties(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        '''
        4000:
            ["mtvema", 67728, 7.17935358987, 0.0903909756674, "-13.0500_-52.2000"]
            ["myfavoritesongsever", 17990, 7.62581779614, 0.0336297943302, "37.7000_-76.8500"]
            ["harrypotterlive", 7932, 6.69493127916, 0.100226928896, "-23.2000_-46.4000"]
        1600:
            ["julywish", 3955, 3.21284681657, 0.498609355247, "-5.8000_105.8500"]
            ["iowadebate", 3345, 6.32738640295, 0.0588938714499, "33.3500_-117.4500"]
            ["pcwindows7", 1559, 2.2814540684, 0.397049390635, "-5.8000_105.8500"]
            ["dvdhp7", 1433, 3.73585831533, 0.287508722959, "-4.3500_-42.0500"]
        800:
            ["supportorhate", 2655, 0.00482746495611, 0.999623352166, "31.9000_-95.7000"]
            ["mileynoacessomtv", 2353, 0.908884671906, 0.841903952401, "-24.6500_-52.2000"]
            ["britneyenovidadepop", 2287, 0.558358402398, 0.890249234805, "-2.9000_-39.1500"]
        '''
        input_file = f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) \
                + GeneralMethods.get_method_id() + '_%s.png'
        output_text_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) \
                + GeneralMethods.get_method_id() + '/%s.txt'
        ltuo_entropy_focus_coverage_hashtag_occurrence_count_and_focus_location = [(data[2], data[3][1], data[4], data[0], data[1], data[3][0]) for data in iterateJsonFromFile(input_file)]
        mf_coverage_to_entropies = defaultdict(list)
        mf_coverage_to_focuses = defaultdict(list)
        mf_coverage_boundary_to_tuo_entropy_and_focus_and_hashtag_and_occurrence_count_and_focus_location = defaultdict(list)
        total_hashtags = len(ltuo_entropy_focus_coverage_hashtag_occurrence_count_and_focus_location)+0.
        for entropy, focus, coverage, hashtag, occurrence_count, focus_location in ltuo_entropy_focus_coverage_hashtag_occurrence_count_and_focus_location:
            coverage = int(coverage/100)*100+100
            mf_coverage_to_entropies[coverage].append(entropy)
            mf_coverage_to_focuses[coverage].append(focus)
            coverage_boundary = 800
            if 800<coverage<1600: coverage_boundary=1600
            elif 1600<coverage: coverage_boundary=4000
            mf_coverage_boundary_to_tuo_entropy_and_focus_and_hashtag_and_occurrence_count_and_focus_location[coverage_boundary].append((entropy, focus, hashtag, occurrence_count, focus_location))
        
        for coverage_boundary, ltuo_entropy_and_focus_and_hashtag_and_occurrence_count_and_focus_location in \
                mf_coverage_boundary_to_tuo_entropy_and_focus_and_hashtag_and_occurrence_count_and_focus_location.iteritems():
            ltuo_entropy_and_focus_and_hashtag_and_s_occurrence_count_and_focus_location = \
                sorted(ltuo_entropy_and_focus_and_hashtag_and_occurrence_count_and_focus_location, key=itemgetter(3), reverse=True)
            for entropy, focus, hashtag, occurrence_count, focus_location in \
                    ltuo_entropy_and_focus_and_hashtag_and_s_occurrence_count_and_focus_location:
                FileIO.writeToFileAsJson([hashtag, occurrence_count, entropy, focus, focus_location], output_text_file%coverage_boundary)
            print coverage_boundary, len(ltuo_entropy_and_focus_and_hashtag_and_occurrence_count_and_focus_location)/total_hashtags
            print 'median entropy: ', np.median(zip(*ltuo_entropy_and_focus_and_hashtag_and_s_occurrence_count_and_focus_location)[0])
            print 'median focus: ', np.median(zip(*ltuo_entropy_and_focus_and_hashtag_and_s_occurrence_count_and_focus_location)[1])
#            print 'var entropy: ', np.var(zip(*ltuo_entropy_and_focus_and_hashtag_and_s_occurrence_count_and_focus_location)[0])
#            print 'var focus: ', np.var(zip(*ltuo_entropy_and_focus_and_hashtag_and_s_occurrence_count_and_focus_location)[1])

#            print 'range entropy: ', getOutliersRangeUsingIRQ(zip(*ltuo_entropy_and_focus_and_hashtag_and_s_occurrence_count_and_focus_location)[0])
#            print 'range focus: ', getOutliersRangeUsingIRQ(zip(*ltuo_entropy_and_focus_and_hashtag_and_s_occurrence_count_and_focus_location)[1])
            
        x_coverages, y_entropies = zip(*[(coverage, np.mean(entropies)) 
                                         for coverage, entropies in mf_coverage_to_entropies.iteritems()
                                         if len(entropies) > 250])
        x_coverages, y_focuses = zip(*[(coverage, np.mean(focuses)) 
                                         for coverage, focuses in mf_coverage_to_focuses.iteritems()
                                         if len(focuses) > 250])
        plt.figure(num=None, figsize=(4.3,3))
        ax = plt.subplot(111)
        plt.subplots_adjust(bottom=0.2, top=0.9, left=0.15)
        plt.scatter(x_coverages, y_entropies, lw=0, marker='o', c='k', s=25)
#        plt.ylim(ymax=1.2)
        plt.xlabel('Spread (miles)')
        plt.ylabel('Entropy')
#        ax.set_xscale('log')
        plt.grid(True)
        savefig(output_file%'entropy')
        
        plt.figure(num=None, figsize=(4.3,3))
        ax = plt.subplot(111)
        plt.subplots_adjust(bottom=0.2, top=0.9, left=0.15)
        plt.scatter(x_coverages, y_focuses, lw=0, marker='o', c='k', s=25)
#        plt.ylim(ymax=1.2)
        plt.xlabel('Spread (miles)')
        plt.ylabel('Focus')
#        ax.set_xscale('log')
        plt.grid(True)
        savefig(output_file%'focus')
        
#        plt.scatter(x_coverages, y_entropies)
#        plt.scatter(x_coverages, y_focuses)
#        plt.show()
    @staticmethod
    def hashtag_stats(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        input_file = f_hashtag_objects%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        hashtags = [len(hashtag['ltuo_lid_and_s_interval']) for hashtag in iterateJsonFromFile(input_file)]
        print 'No. of hashtags: ', len(hashtags)
        print 'Total no. of occurrences: ', sum(hashtags)
    @staticmethod
    def peak_categories(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        def get_coverage_boundary(coverage):
            coverage = int(coverage/100)*100+100
            coverage_boundary = 800
            if 800<coverage<1600: coverage_boundary=1600
            elif 1600<coverage: coverage_boundary=4000
            return coverage_boundary
        def print_coverage_stats(ltuo_coverage_and_peak):
            coverages = zip(*ltuo_coverage_and_peak)[0]
            total_coverages = len(coverages)+0.
            coverage_800 = len([coverage for coverage in coverages if coverage==800])
            coverage_1600 = len([coverage for coverage in coverages if coverage==1600])
            coverage_4000 = len([coverage for coverage in coverages if coverage==4000])
            print '%0.2f'%(coverage_800/total_coverages), '%0.2f'%(coverage_1600/total_coverages), '%0.2f'%(coverage_4000/total_coverages)
        input_file = f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        ltuo_coverage_and_peak = [(data[4], data[5]) for data in iterateJsonFromFile(input_file)]
        total_hashtags = len(ltuo_coverage_and_peak)+0.
        ltuo_coverage_and_peak_lt_6 = [(get_coverage_boundary(coverage), peak) for coverage, peak in ltuo_coverage_and_peak if peak<=6]
        ltuo_coverage_and_peak_gt_6_lt_120 = [(get_coverage_boundary(coverage), peak) for coverage, peak in ltuo_coverage_and_peak if peak>6 and peak<=120]
        ltuo_coverage_and_peak_gt_120 = [(get_coverage_boundary(coverage), peak) for coverage, peak in ltuo_coverage_and_peak if peak>120]
        print 'lt 30: %0.2f'%(len(ltuo_coverage_and_peak_lt_6)/total_hashtags)
        print_coverage_stats(ltuo_coverage_and_peak_lt_6)
        print 'gt 30, lt 600: %0.2f'%(len(ltuo_coverage_and_peak_gt_6_lt_120)/total_hashtags)
        print_coverage_stats(ltuo_coverage_and_peak_gt_6_lt_120)
        print 'gt 600: %0.2f'%(len(ltuo_coverage_and_peak_gt_120)/total_hashtags)
        print_coverage_stats(ltuo_coverage_and_peak_gt_120)
    @staticmethod
    def run():
#        input_files_start_time, input_files_end_time, min_no_of_hashtags = datetime(2011, 2, 1), datetime(2011, 2, 27), 0
        input_files_start_time, input_files_end_time, min_no_of_hashtags = datetime(2011, 2, 1), datetime(2012, 4, 30), 50
        
#        DataAnalysis.hashtag_distribution(input_files_start_time, input_files_end_time)
#        DataAnalysis.occurrence_distribution_by_country(input_files_start_time, input_files_end_time)
#        DataAnalysis.fraction_of_occurrences_vs_rank_of_country(input_files_start_time, input_files_end_time)
        
#        DataAnalysis.occurrence_distribution_by_country(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.occurrence_distribution_by_world_map(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.hashtag_distribution_loglog(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.hashtag_locations_distribution_loglog(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.fraction_of_occurrences_vs_rank_of_location(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.cumulative_fraction_of_occurrences_vs_rank_of_location(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.top_k_locations_on_world_map(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.write_entropy_and_focus(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.write_top_locations(input_files_start_time, input_files_end_time, min_no_of_hashtags)

#        DataAnalysis.locality_measure_cdf(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.locality_measures_vs_nuber_of_occurreneces(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.ef_plot(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.locality_measures_locality_specific_correlation(input_files_start_time, input_files_end_time, min_no_of_hashtags, plot_country=False)    
#        DataAnalysis.locality_measures_location_specific_correlation_example_hashtags(input_files_start_time, input_files_end_time, min_no_of_hashtags, plot_country=False  )

#        DataAnalysis.iid_vs_cumulative_distribution_and_peak_distribution(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.peak_stats(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.occurrence_decay(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.norm_iid_vs_locality_measuers(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.ef_plots_for_peak(input_files_start_time, input_files_end_time, min_no_of_hashtags)

#        DataAnalysis.peak_lids_dist(input_files_start_time, input_files_end_time, min_no_of_hashtags)        
#        DataAnalysis.distance_vs_no_of_common_hashtags(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.distance_vs_temporal_distance(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.write_examples_of_locations_at_different_ends_of_distance_spectrum(input_files_start_time, input_files_end_time, min_no_of_hashtags)
        
#        DataAnalysis.cumulative_fraction_of_occurrences_vs_rank_of_country(input_files_start_time, input_files_end_time)
#        DataAnalysis.coverage_vs_spatial_properties(input_files_start_time, input_files_end_time, min_no_of_hashtags)
        
#        DataAnalysis.hashtag_stats(input_files_start_time, input_files_end_time, min_no_of_hashtags)
        DataAnalysis.peak_categories(input_files_start_time, input_files_end_time, min_no_of_hashtags)

class LocationRelationshipAnalysis():
    @staticmethod
    def sharing_analysis(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        input_file = f_hashtag_objects%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
#        input_file = 'data/hashtag_objects'
        output_file = f_tuo_valid_focus_lid_pair_and_common_hashtag_affinity_score%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        GeneralMethods.runCommand('rm -rf %s'%output_file)
        mf_focus_lid_to_so_hashtags = defaultdict(set)
        mf_valid_focus_lid_pair_to_affinity_score = {}
        valid_focus_lid_pairs = set()
        for hashtag_count, hashtag_object in enumerate(iterateJsonFromFile(input_file)): 
            print hashtag_count
            so_observed_focus_lids = get_so_observed_focus_lids(hashtag_object)
            for focus_lid in so_observed_focus_lids: mf_focus_lid_to_so_hashtags[focus_lid].add(hashtag_object['hashtag'])
            for focus_lid1, focus_lid2 in combinations(so_observed_focus_lids,2): valid_focus_lid_pairs.add(':ilab:'.join(sorted([focus_lid1, focus_lid2])))
        for valid_focus_lid_pair in valid_focus_lid_pairs:
            focus_lid1, focus_lid2 = valid_focus_lid_pair.split(':ilab:')
            so_hashtags_for_focus_lid1 = mf_focus_lid_to_so_hashtags[focus_lid1]
            so_hashtags_for_focus_lid2 = mf_focus_lid_to_so_hashtags[focus_lid2]
            mf_valid_focus_lid_pair_to_affinity_score[valid_focus_lid_pair] = len(so_hashtags_for_focus_lid1.intersection(so_hashtags_for_focus_lid2))/(len(so_hashtags_for_focus_lid1.union(so_hashtags_for_focus_lid2))+0.)
        for k, v in mf_valid_focus_lid_pair_to_affinity_score.iteritems(): FileIO.writeToFileAsJson([k,v], output_file)
    @staticmethod
    def temporal_analysis(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        input_file = f_hashtag_objects%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
#        input_file = 'data/hashtag_objects'
        output_file = f_tuo_valid_focus_lid_pair_and_temporal_affinity_score%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        GeneralMethods.runCommand('rm -rf %s'%output_file)
        mf_focus_lid_to_mf_hashtag_to_occurrence_iid = defaultdict(dict)
        mf_valid_focus_lid_pair_to_affinity_score = {}
        valid_focus_lid_pairs = set()
        for hashtag_count, hashtag_object in enumerate(iterateJsonFromFile(input_file)): 
            print hashtag_count
            ltuo_valid_iid_and_focus_lid = get_ltuo_valid_iid_and_focus_lid(hashtag_object)
            so_observed_focus_lids = set(zip(*ltuo_valid_iid_and_focus_lid)[1])
            for valid_iid, focus_lid in ltuo_valid_iid_and_focus_lid: mf_focus_lid_to_mf_hashtag_to_occurrence_iid[focus_lid][hashtag_object['hashtag']] = valid_iid
            for focus_lid1, focus_lid2 in combinations(so_observed_focus_lids,2): valid_focus_lid_pairs.add(':ilab:'.join(sorted([focus_lid1, focus_lid2])))
        for valid_focus_lid_pair in valid_focus_lid_pairs:
            focus_lid1, focus_lid2 = valid_focus_lid_pair.split(':ilab:')
            mf_hashtag_to_occurrence_iid1 = mf_focus_lid_to_mf_hashtag_to_occurrence_iid[focus_lid1]
            mf_hashtag_to_occurrence_iid2 = mf_focus_lid_to_mf_hashtag_to_occurrence_iid[focus_lid2]
            common_hashtags = set(mf_hashtag_to_occurrence_iid1.keys()).intersection(set(mf_hashtag_to_occurrence_iid2.keys()))
            affinity_scores = [
                                   abs(mf_hashtag_to_occurrence_iid2[hashtag]-mf_hashtag_to_occurrence_iid1[hashtag]) 
                                   for hashtag in common_hashtags
                               ]
            mf_valid_focus_lid_pair_to_affinity_score[valid_focus_lid_pair] = np.mean(affinity_scores)
        for k, v in mf_valid_focus_lid_pair_to_affinity_score.iteritems(): FileIO.writeToFileAsJson([k,v], output_file)
    @staticmethod
    def _plot_affinities(input_file, type=None):
        ltuo_valid_focus_lid_pair_and_affinity_score = [data for data in iterateJsonFromFile(input_file)]
        mf_distance_to_affinity_scores = defaultdict(list)
        for valid_focus_lid_pair, affinity_score in ltuo_valid_focus_lid_pair_and_affinity_score:
            lid1, lid2 = valid_focus_lid_pair.split(':ilab:')
            distance = getHaversineDistance(getLocationFromLid(lid1.replace('_', ' ')), getLocationFromLid(lid2.replace('_', ' ')))
            distance=int(distance/100)*100+100
            mf_distance_to_affinity_scores[distance].append(affinity_score)
        ltuo_distance_and_affinity_score = [(distance, np.mean(affinity_scores)) for distance, affinity_scores in mf_distance_to_affinity_scores.iteritems() if len(affinity_scores)>100]
        x_distances, y_affinity_scores = zip(*sorted(ltuo_distance_and_affinity_score, key=itemgetter(0)))
        if type=='temporal': y_affinity_scores = [y*TIME_UNIT_IN_SECONDS/(60.*60.) for y in y_affinity_scores]
#        total_occurrences = sum(mf_distance_to_total_co_occurrences.values())
#        x_distance, y_total_co_occurrences = zip(*sorted(mf_distance_to_total_co_occurrences.items(), key=itemgetter(0)))
#        y_total_co_occurrences = [y/total_occurrences for y in y_total_co_occurrences]
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9, wspace=0, hspace=0)
        x_distances, y_affinity_scores = splineSmooth(x_distances, y_affinity_scores)
        plt.semilogx(x_distances, y_affinity_scores, c='k', lw=2)
#        plt.plot(x_distances, y_affinity_scores, c='k', lw=2)
        plt.xlim(xmin=95, xmax=15000)
        plt.grid(True)
    @staticmethod
    def content_affinity_vs_distance(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        input_file = f_tuo_valid_focus_lid_pair_and_common_hashtag_affinity_score%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) \
                + GeneralMethods.get_method_id() + '.png'
        LocationRelationshipAnalysis._plot_affinities(input_file)
        plt.xlabel('Distance (miles)')
        plt.ylabel('Hashtags sharing similarity')
#        plt.show()
        savefig(output_file)
    @staticmethod
    def temporal_affinity_vs_distance(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        input_file = f_tuo_valid_focus_lid_pair_and_temporal_affinity_score%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) \
                + GeneralMethods.get_method_id() + '.png'
        LocationRelationshipAnalysis._plot_affinities(input_file, type='temporal')
        plt.xlabel('Distance (miles)')
        plt.ylabel('Hashtag adoption lag (hours)')
#        plt.show()
        savefig(output_file)
    @staticmethod
    def run():
        input_files_start_time, input_files_end_time, min_no_of_hashtags = datetime(2011, 2, 1), datetime(2012, 4, 30), 50
#        LocationRelationshipAnalysis.sharing_analysis(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        LocationRelationshipAnalysis.temporal_analysis(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        LocationRelationshipAnalysis.content_affinity_vs_distance(input_files_start_time, input_files_end_time, min_no_of_hashtags)
        LocationRelationshipAnalysis.temporal_affinity_vs_distance(input_files_start_time, input_files_end_time, min_no_of_hashtags)

        
if __name__ == '__main__':
    DataAnalysis.run()
#    LocationRelationshipAnalysis.run()
#    CountryBoundaries.run()
    