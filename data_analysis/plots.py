'''
Created on May 8, 2012

@author: krishnakamath
'''

from datetime import datetime
from settings import f_tuo_normalized_occurrence_count_and_distribution_value,\
    fld_sky_drive_data_analysis_images, f_tuo_lid_and_distribution_value,\
    f_tuo_rank_and_average_percentage_of_occurrences, \
    f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak, \
    f_tuo_iid_and_interval_stats
from library.file_io import FileIO
from library.classes import GeneralMethods
from operator import itemgetter
import matplotlib.pyplot as plt
from library.plotting import savefig, splineSmooth
import shapefile, os
from library.geo import point_inside_polygon, getLocationFromLid
from collections import defaultdict
from library.stats import entropy, focus
import numpy as np
import matplotlib
from datetime import timedelta
from mr_analysis import TIME_UNIT_IN_SECONDS
from data_analysis.settings import f_tuo_normalized_iid_and_tuo_prct_of_occurrences_and_entropy_and_focus_and_coverage

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
        plt.plot(x_percentage_of_locations, y_average_percentage_of_occurrences, lw=0, marker='>')  
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
            plt.scatter(x_occurrance_counts, y_locality_measures)
            plt.semilogx(x_occurrance_counts[0], y_locality_measures[0])
            savefig(output_file%id)
        ACCURACY_NO_OF_OCCURRANCES = 25
        input_file = f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
        ltuo_hashtag_and_occurrence_count_and_entropy_and_focus = [data for data in iterateJsonFromFile(input_file)]
        ltuo_entropy_and_occurrences_count = [(data[2], data[1]) for data in ltuo_hashtag_and_occurrence_count_and_entropy_and_focus]
        ltuo_focus_and_occurrences_count = [(data[3][1], data[1]) for data in ltuo_hashtag_and_occurrence_count_and_entropy_and_focus]
        plot_graph(ltuo_entropy_and_occurrences_count, 'entropy')
        plot_graph(ltuo_focus_and_occurrences_count, 'focus')
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
            plt.plot(x_measure, y_distribution, lw=0, marker='o')
            savefig(output_file%id)
        input_file = f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), no_of_hashtags)
        ltuo_hashtag_and_occurrence_count_and_entropy_and_focus = [data for data in iterateJsonFromFile(input_file)]
        entropies = zip(*ltuo_hashtag_and_occurrence_count_and_entropy_and_focus)[2]
        focuses = zip(*ltuo_hashtag_and_occurrence_count_and_entropy_and_focus)[3]
        focuses = zip(*focuses)[1]
        print 'Mean entropy: ', np.mean(entropies)
        print 'Mean focus: ', np.mean(focuses)
        plot_graph(entropies, 'entropy')
        plot_graph(focuses, 'focus')
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
                plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
                x_focus, y_entropy = zip(*[(norm_focus, np.mean(entropies)) for norm_focus, entropies in mf_norm_focus_to_entropies.iteritems() if len(entropies)>5])
                _, z_coverage = zip(*[(norm_focus, np.mean(coverages)) for norm_focus, coverages in mf_norm_focus_to_coverages.iteritems() if len(coverages)>5])
                cm = matplotlib.cm.get_cmap('autumn')
                sc = plt.scatter(x_focus, y_entropy, c=z_coverage, cmap=cm, s=50, lw=0, vmin=0, vmax=2990)
                plt.xlim(xmin=-0.1, xmax=1.1)
                plt.ylim(ymin=-1, ymax=9)
                plt.colorbar(sc)
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
        plt.semilogx(x_iids, y_is_peaks, marker='o')
        plt.xlabel('Minutes')
        savefig(output_file%'peaks')
        plt.clf()
        plt.plot(x_iids, z_cumulative_percentage_of_occurrencess, lw=0, marker='o')
        plt.xlabel('Minutes')
#        savefig(output_file%'cumulative_percentage_of_occurrences')
        plt.show()
    @staticmethod
    def norm_iid_vs_locality_measuers(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        input_file = f_tuo_normalized_iid_and_tuo_prct_of_occurrences_and_entropy_and_focus_and_coverage%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) \
                + GeneralMethods.get_method_id() + '/%s.png'
        ltuo_normalized_iid_and_tuo_prct_of_occurrences_and_entropy_and_focus_and_coverage = \
            [data for data in iterateJsonFromFile(input_file)]
#        for data in ltuo_normalized_iid_and_tuo_prct_of_occurrences_and_entropy_and_focus_and_coverage:
#            print data
        x_normalized_iids, y_entropies, y_focuses = zip(*sorted([(data[0],data[1][1], data[1][2]) 
                                                                      for data in 
                                                                        ltuo_normalized_iid_and_tuo_prct_of_occurrences_and_entropy_and_focus_and_coverage
                                                                  ])
                                                        )
        
        plt.xlim(xmin=-24, xmax=100)               
        plt.plot(x_normalized_iids, y_entropies, lw=0, marker='o')                               
        plt.show()                     
        
#        mf_peak_to_plot_data = defaultdict(dict)
#        for data in sorted(ltuo_normalized_iid_and_tuo_prct_of_occurrences_and_entropy_and_focus_and_coverage, key=itemgetter(0)):
#            peak = None
#            if '_' not in str(data[0]): iid, peak = int(data[0]), -1                
#            else: 
#                iid, peak = data[0].split('_')
#                iid, peak = int(iid), int(peak)
#            if peak < 5:
#                if peak not in mf_peak_to_plot_data: mf_peak_to_plot_data[peak] = defaultdict(list)
#                mf_peak_to_plot_data[peak]['x_normalized_iids'].append(iid)
#                mf_peak_to_plot_data[peak]['y_entropies'].append(data[1][1])
#                mf_peak_to_plot_data[peak]['y_focuses'].append(data[1][2])
#                mf_peak_to_plot_data[peak]['z_coverages'].append(data[1][3])
            
#        plt.plot(mf_peak_to_plot_data[-1]['x_normalized_iids'], mf_peak_to_plot_data[-1]['y_entropies'])
#        xmin=-2 
#        xmax=24
#        for peak, plot_data in mf_peak_to_plot_data.iteritems():
#            if peak==-1:
#                x_normalized_iids, y_focuses = [], []
#                for x, y in zip(plot_data['x_normalized_iids'], plot_data['y_focuses']):
##                    if x>=xmin and x<=xmax:
#                        x_normalized_iids.append(x)
#                        y_focuses.append(y)
#                print len(x_normalized_iids), len(y_focuses)
##                x_normalized_iids, y_focuses = splineSmooth(x_normalized_iids, y_focuses)
#                plt.plot(x_normalized_iids, y_focuses, lw=0, marker='o')
##                plt.semilogx()
#        plt.xlim(xmin=-24, xmax=100)
##        plt.ylim(ymin=0.68, ymax=0.9)
#        plt.legend()
#        plt.show()
    @staticmethod
    def locality_measures_correlation_with_peak(input_files_start_time, input_files_end_time, min_no_of_hashtags):
        def getNearestNumber(num): return  (int(round(num,2)*100/100)*100 + int((round(num,2)*100%100)/3)*3)/100.
        output_file = \
                fld_sky_drive_data_analysis_images%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags) \
                + GeneralMethods.get_method_id() + '/%s/%s.png'
        input_file = f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_no_of_hashtags)
        ltuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak = [data for data in iterateJsonFromFile(input_file)]
        print len(ltuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak)
        _, _, entropies, focuses, _, peaks = zip(*ltuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak)
        focuses = zip(*focuses)[1]
        mf_norm_focus_to_entropies = defaultdict(list)
        mf_norm_focus_to_peaks = defaultdict(list)
        mf_norm_focus_to_mf_peak_to_values = defaultdict(dict)
        for focus, entropy, peak in zip(focuses,entropies, peaks):
            if peak<150:
                mf_norm_focus_to_entropies[round(focus, 2)].append(entropy)
                mf_norm_focus_to_peaks[round(focus, 2)].append(peak)
                
#                mf_norm_focus_to_entropies[getNearestNumber(focus)].append(entropy)
#                mf_norm_focus_to_peaks[getNearestNumber(focus)].append(peak)

#                peak = int(peak/10)*10+10
#                if peak not in mf_norm_focus_to_mf_peak_to_values: mf_norm_focus_to_mf_peak_to_values[peak]=[]
#                mf_norm_focus_to_mf_peak_to_values[peak].append(peak)
#        plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
        x_focus, y_entropy = zip(*[(norm_focus, np.mean(entropies)) for norm_focus, entropies in mf_norm_focus_to_entropies.iteritems() if len(entropies)>5])
        _, z_peak = zip(*[(norm_focus, np.mean(peaks)*TIME_UNIT_IN_SECONDS/60) for norm_focus, peaks in mf_norm_focus_to_peaks.iteritems() if len(peaks)>5])
#        z_peak = [int(peak/10)*10+10 for peak in z_peak]
        cm = matplotlib.cm.get_cmap('cool')
        print len(x_focus)
        x_focus, y_entropy, z_peak = zip(*sorted(zip(x_focus, y_entropy, z_peak), key=itemgetter(2),reverse=True))
        sc = plt.scatter(x_focus, y_entropy, c=z_peak, cmap=cm, s=50, lw=0,)
#        plt.xlim(xmin=-0.1, xmax=1.1)
#        plt.ylim(ymin=-1, ymax=9)
        plt.colorbar(sc)
#        if plot_country: savefig(output_file%('country', country))
#        else: savefig(output_file%('location', country))
        plt.show()
#        plt.scatter(x_focus, z_peak)
#        plt.show()
        plt.clf()
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

#        DataAnalysis.locality_measure_cdf(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.locality_measures_vs_nuber_of_occurreneces(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.locality_measures_locality_specific_correlation(input_files_start_time, input_files_end_time, min_no_of_hashtags, plot_country=False)    
#        DataAnalysis.locality_measures_location_specific_correlation_example_hashtags(input_files_start_time, input_files_end_time, min_no_of_hashtags, plot_country=False  )

#        DataAnalysis.iid_vs_cumulative_distribution_and_peak_distribution(input_files_start_time, input_files_end_time, min_no_of_hashtags)
#        DataAnalysis.norm_iid_vs_locality_measuers(input_files_start_time, input_files_end_time, min_no_of_hashtags)
        DataAnalysis.locality_measures_correlation_with_peak(input_files_start_time, input_files_end_time, min_no_of_hashtags)
        
#        DataAnalysis.cumulative_fraction_of_occurrences_vs_rank_of_country(input_files_start_time, input_files_end_time)
        
if __name__ == '__main__':
    DataAnalysis.run()
#    CountryBoundaries.run()
    