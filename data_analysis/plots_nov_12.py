'''
Created on Nov 14, 2012

@author: krishnakamath
'''
from collections import defaultdict
from itertools import groupby
from library.classes import GeneralMethods
from library.file_io import FileIO
from library.geo import UTMConverter
from library.geo import plotPointsOnWorldMap
from library.plotting import savefig
from library.plotting import splineSmooth
from library.stats import filter_outliers
from operator import itemgetter
from settings import f_dense_hashtag_distribution_in_locations
from settings import f_dense_hashtags_similarity_and_lag
from settings import f_hashtag_and_location_distribution
from settings import f_hashtag_spatial_metrics
from settings import f_iid_spatial_metrics
from settings import f_norm_iid_spatial_metrics
from settings import fld_data_analysis_results
import matplotlib
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
#        for i, d in enumerate(ltuo_lid_and_r_occurrence_count):
#            print i, d
#        exit()
        lids = zip(*ltuo_lid_and_r_occurrence_count)[0][:200]
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
        ltuo_distance_and_num_samples = [(distance, affinity_scores) for distance, affinity_scores in mf_distance_to_affinity_scores.iteritems()]
        ltuo_distance_and_num_samples.sort(key=itemgetter(0))
#        for distance, num_samples in ltuo_distance_and_num_samples:
#            print distance, len(num_samples), np.mean(num_samples), np.mean(filter_outliers(num_samples))
#        exit()
        ltuo_distance_and_affinity_score = [(distance, np.mean(filter_outliers(affinity_scores))) 
                                            for distance, affinity_scores in mf_distance_to_affinity_scores.iteritems()
                                                if len(affinity_scores)>100]
        x_distances, y_affinity_scores = zip(*sorted(ltuo_distance_and_affinity_score, key=itemgetter(0)))
        if type=='adoption_lag': 
            y_affinity_scores = [y/(60.*60.*60) for y in y_affinity_scores]
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
    def spatial_metrics_vs_occurrence_count():
        output_file_format = fld_data_analysis_results%GeneralMethods.get_method_id()+'/%s.png'
        def plot_graph(ltuo_locality_measure_and_occurrences_count, id):
            mf_normalized_occurrences_count_to_locality_measures = defaultdict(list)
            for locality_measure, occurrences_count in \
                    ltuo_locality_measure_and_occurrences_count:
                normalized_occurrence_count =\
                int(occurrences_count/ACCURACY_NO_OF_OCCURRANCES)*ACCURACY_NO_OF_OCCURRANCES+ACCURACY_NO_OF_OCCURRANCES
                mf_normalized_occurrences_count_to_locality_measures[normalized_occurrence_count].append(
                                                                                                        locality_measure
                                                                                                    )
            x_occurrance_counts, y_locality_measures = [], []
            for k in sorted(mf_normalized_occurrences_count_to_locality_measures):
                if len(mf_normalized_occurrences_count_to_locality_measures[k]) > 10:
                    x_occurrance_counts.append(k), y_locality_measures.append(
                                                     np.mean(mf_normalized_occurrences_count_to_locality_measures[k])
                                                    )
            x_occurrance_counts = [x/1000. for x in x_occurrance_counts]
            plt.figure(num=None, figsize=(4.3,3.0))
            plt.subplots_adjust(bottom=0.2, top=0.9, left=0.15, wspace=0.)
            plt.scatter(x_occurrance_counts, y_locality_measures, lw=0, marker='o', c='k', s=50)
            plt.xlabel('Hashtag occurrences in thousands')
            plt.ylabel('Mean hashtag %s'%id)
            plt.grid(True)
            savefig(output_file_format%('locality_vs_occurrences_'+id))
        ACCURACY_NO_OF_OCCURRANCES = 25
#        import matplotlib as mpl
#        mpl.rcParams['text.usetex']=True
        data = [d for d in FileIO.iterateJsonFromFile(f_hashtag_spatial_metrics, remove_params_dict=True)]
        ltuo_entropy_and_occurrences_count = map(itemgetter('entropy', 'num_of_occurrenes'), data)
        ltuo_focus_and_occurrences_count = map(itemgetter('focus', 'num_of_occurrenes'), data)
        ltuo_focus_and_occurrences_count = [(s, c) for ((_, s), c) in ltuo_focus_and_occurrences_count]
        ltuo_coverage_and_occurrences_count = map(itemgetter('spread', 'num_of_occurrenes'), data)
        plot_graph(ltuo_entropy_and_occurrences_count, 'entropy')
        plot_graph(ltuo_focus_and_occurrences_count, 'focus')
        plot_graph(ltuo_coverage_and_occurrences_count, 'spread')
    @staticmethod
    def spatial_metrics_cdf():
        output_file_format = fld_data_analysis_results%GeneralMethods.get_method_id()+'/%s.png'
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
            savefig(output_file_format%('cdf_'+id))
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
            savefig(output_file_format%('cdf_'+id))
        data = [d for d in FileIO.iterateJsonFromFile(f_hashtag_spatial_metrics, remove_params_dict=True)]
        focuses = map(itemgetter(1), map(itemgetter('focus'), data))
        entropies = map(itemgetter('entropy'), data)
        coverages = map(itemgetter('spread'), data)
        print 'Mean entropy: ', np.mean(entropies)
        print 'Mean focus: ', np.mean(focuses)
        print 'Median entropy: ', np.median(entropies)
        print 'Median focus: ', np.median(focuses)
        plot_graph(focuses, 'Focus')
        plot_graph(entropies, 'Entropy')
        plot_coverage(coverages, 'Spread')
    @staticmethod
    def ef_plot():
        output_file = fld_data_analysis_results%GeneralMethods.get_method_id()+'.png'
        data = [d for d in FileIO.iterateJsonFromFile(f_hashtag_spatial_metrics, remove_params_dict=True)]
        ltuo_hashtag_and_entropy_and_focus = map(itemgetter('hashtag', 'entropy', 'focus'), data)
        mf_norm_focus_to_entropies = defaultdict(list)
        for _, entropy, (_, focus) in ltuo_hashtag_and_entropy_and_focus:
            mf_norm_focus_to_entropies[round(focus, 2)].append(entropy)
        plt.figure(num=None, figsize=(6,3))
        x_focus, y_entropy = zip(*[(norm_focus, np.mean(entropies))
                                    for norm_focus, entropies in mf_norm_focus_to_entropies.iteritems()
                                    if len(entropies)>0])
        plt.subplots_adjust(bottom=0.2, top=0.9, wspace=0, hspace=0)
        plt.scatter(x_focus, y_entropy, s=50, lw=0, c='k')
        plt.xlim(xmin=-0.1, xmax=1.1)
        plt.ylim(ymin=-1, ymax=9)
        plt.xlabel('Mean hashtag focus')
        plt.ylabel('Mean hashtag entropy')
        plt.grid(True)
        savefig(output_file)
        ltuo_hashtag_and_r_entropy_and_focus =\
                                            sorted(ltuo_hashtag_and_entropy_and_focus, key=itemgetter(1), reverse=True)
        ltuo_hashtag_and_r_entropy_and_s_focus = sorted(ltuo_hashtag_and_r_entropy_and_focus, key=itemgetter(2))
        hashtags = zip(*ltuo_hashtag_and_r_entropy_and_s_focus)[0]
        print list(hashtags[:20])
        print list(reversed(hashtags))[:20]
    @staticmethod
    def coverage_vs_spatial_properties():
        output_file_format = fld_data_analysis_results%GeneralMethods.get_method_id()+'_%s.png'
        output_text_file_format = fld_data_analysis_results%GeneralMethods.get_method_id()+'/%s.txt'
        data = [d for d in FileIO.iterateJsonFromFile(f_hashtag_spatial_metrics, remove_params_dict=True)]
        keys = ['entropy', 'focus', 'spread', 'hashtag', 'num_of_occurrenes', 'focus']
        ltuo_entropy_focus_coverage_hashtag_occurrence_count_and_focus_location = map(itemgetter(*keys), data)
        ltuo_entropy_focus_coverage_hashtag_occurrence_count_and_focus_location =\
                                        map(
                                              lambda (a,b,c,d,e,f): (a,b[1],c,d,e,f[0]),
                                              ltuo_entropy_focus_coverage_hashtag_occurrence_count_and_focus_location
                                        )
#        ltuo_entropy_focus_coverage_hashtag_occurrence_count_and_focus_location = [(data[2], data[3][1], data[4], data[0], data[1], data[3][0]) for data in iterateJsonFromFile(input_file)]
        mf_coverage_to_entropies = defaultdict(list)
        mf_coverage_to_focuses = defaultdict(list)
        mf_coverage_boundary_to_tuo_entropy_and_focus_and_hashtag_and_occurrence_count_and_focus_location =\
                                                                                                    defaultdict(list)
        total_hashtags = len(ltuo_entropy_focus_coverage_hashtag_occurrence_count_and_focus_location)+0.
        for entropy, focus, coverage, hashtag, occurrence_count, focus_location in\
                ltuo_entropy_focus_coverage_hashtag_occurrence_count_and_focus_location:
            coverage = int(coverage/100)*100+100
            mf_coverage_to_entropies[coverage].append(entropy)
            mf_coverage_to_focuses[coverage].append(focus)
            coverage_boundary = 800
            if 800<coverage<1600: coverage_boundary=1600
            elif 1600<coverage: coverage_boundary=4000
            mf_coverage_boundary_to_tuo_entropy_and_focus_and_hashtag_and_occurrence_count_and_focus_location\
                                [coverage_boundary].append((entropy, focus, hashtag, occurrence_count, focus_location))
        
        for coverage_boundary, ltuo_entropy_and_focus_and_hashtag_and_occurrence_count_and_focus_location in \
                mf_coverage_boundary_to_tuo_entropy_and_focus_and_hashtag_and_occurrence_count_and_focus_location\
                                                                                                        .iteritems():
            ltuo_entropy_and_focus_and_hashtag_and_s_occurrence_count_and_focus_location = \
                sorted(ltuo_entropy_and_focus_and_hashtag_and_occurrence_count_and_focus_location, key=itemgetter(3), reverse=True)
            for entropy, focus, hashtag, occurrence_count, focus_location in \
                    ltuo_entropy_and_focus_and_hashtag_and_s_occurrence_count_and_focus_location:
                FileIO.writeToFileAsJson(
                                         [hashtag, occurrence_count, entropy, focus, focus_location],
                                         output_text_file_format%coverage_boundary
                                        )
            print coverage_boundary,\
                        len(ltuo_entropy_and_focus_and_hashtag_and_occurrence_count_and_focus_location)/total_hashtags
            print 'median entropy: ',\
                        np.median(zip(*ltuo_entropy_and_focus_and_hashtag_and_s_occurrence_count_and_focus_location)[0])
            print 'median focus: ',\
                        np.median(zip(*ltuo_entropy_and_focus_and_hashtag_and_s_occurrence_count_and_focus_location)[1])
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
        savefig(output_file_format%'entropy')
        
        plt.figure(num=None, figsize=(4.3,3))
        ax = plt.subplot(111)
        plt.subplots_adjust(bottom=0.2, top=0.9, left=0.15)
        plt.scatter(x_coverages, y_focuses, lw=0, marker='o', c='k', s=25)
#        plt.ylim(ymax=1.2)
        plt.xlabel('Spread (miles)')
        plt.ylabel('Focus')
#        ax.set_xscale('log')
        plt.grid(True)
        savefig(output_file_format%'focus')
    @staticmethod
    def peak_stats():
        TIME_UNIT_IN_SECONDS = 10.*60.
        output_file_format = fld_data_analysis_results%GeneralMethods.get_method_id()+'/%s.png'
        data = [d for d in FileIO.iterateJsonFromFile(f_hashtag_spatial_metrics, remove_params_dict=True)]
        peaks = map(itemgetter('peak_iid'), data)
        peaks = filter(lambda i: i<288, peaks)
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
        savefig(output_file_format%'peak_cdf')
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
        savefig(output_file_format%'peak_dist')
    @staticmethod
    def iid_vs_cumulative_distribution_and_peak_distribution():
        TIME_UNIT_IN_SECONDS = 10.*60.
        output_file_format = fld_data_analysis_results%GeneralMethods.get_method_id()+'/%s.png'
        ltuo_iid_and_interval_stats = [data for data in 
                                        FileIO.iterateJsonFromFile(f_iid_spatial_metrics, remove_params_dict=True)]
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
        savefig(output_file_format%'peaks')
        plt.clf()
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9, wspace=0, hspace=0)
        plt.plot(x_iids, z_cumulative_percentage_of_occurrencess, lw=0, marker='o', c='k')
#        plt.xlabel('Minutes')
        plt.ylabel('CDF of occurrences')
        plt.xlabel('Time (Minutes)')
        plt.grid(True)
        plt.xlim(xmax=600)
        savefig(output_file_format%'cdf_occurrences_peak')
    @staticmethod
    def ef_plots_for_peak():
        output_file_format = fld_data_analysis_results%GeneralMethods.get_method_id()+'/%s.png'
        def getNearestNumber(num): return  (int(round(num,2)*100/100)*100 + int((round(num,2)*100%100)/3)*3)/100.
        def plot_correlation_ef_plot(condition, id, hashtags, focuses, entropies, peaks):
            TIME_UNIT_IN_SECONDS = 10.*60.
            mf_norm_focus_to_entropies = defaultdict(list)
            mf_norm_focus_to_peaks = defaultdict(list)
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
            savefig(output_file_format%id)
            ltuo_hashtag_and_entropy_and_focus = zip(hashtags, entropies, focuses)
            ltuo_hashtag_and_r_entropy_and_focus = sorted(ltuo_hashtag_and_entropy_and_focus, key=itemgetter(1), reverse=True)
            ltuo_hashtag_and_r_entropy_and_s_focus = sorted(ltuo_hashtag_and_r_entropy_and_focus, key=itemgetter(2))
            hashtags = zip(*ltuo_hashtag_and_r_entropy_and_s_focus)[0]
            print id, list(hashtags)
            print id, list(reversed(hashtags))
        data = [d for d in FileIO.iterateJsonFromFile(f_hashtag_spatial_metrics, remove_params_dict=True)]
        hashtags = map(itemgetter('hashtag'), data)
        focuses = map(itemgetter(1), map(itemgetter('focus'), data))
        entropies = map(itemgetter('entropy'), data)
        peaks = map(itemgetter('peak_iid'), data)
        def gt_288(peak):
            if 288>peak and peak<1008: return True
        def lt_6(peak):
            if peak < 6: return True
        def lt_144(peak):
            if peak < 144: return True
        plot_correlation_ef_plot(gt_288, 'gt_288', hashtags, focuses, entropies, peaks)
        plot_correlation_ef_plot(lt_6, 'lt_6', hashtags, focuses, entropies, peaks)
    @staticmethod
    def norm_iid_vs_locality_measuers():
        TIME_UNIT_IN_SECONDS = 10.*60.
        output_file_format = fld_data_analysis_results%GeneralMethods.get_method_id()+'/%s.png'
        ltuo_normalized_iid_and_tuo_prct_of_occurrences_and_entropy_and_focus_and_coverage = \
                    [data for data in FileIO.iterateJsonFromFile(f_norm_iid_spatial_metrics, remove_params_dict=True)]
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
        savefig(output_file_format%'coverage')
        plt.clf() 
        
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9)
        plt.subplot(111)
        plt.xlim(xmin=-20, xmax=120)
        plt.ylim(ymin=0.55, ymax=0.70)
        plt.plot(x_normalized_iids, y_entropies,  lw=1, c='k')
        plt.scatter(x_normalized_iids, y_entropies, lw=0, marker='o', s=50, c='k')
        plt.ylabel('Interval entropy')
        plt.xlabel('Minutes since peak')
        plt.grid(True)
        savefig(output_file_format%'entropy')
        plt.clf() 
        
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9)
        plt.subplot(111)
        plt.xlim(xmin=-20, xmax=400)
#        plt.ylim(ymin=1, ymax=3)
        plt.plot(x_normalized_iids, y_distance_from_overall_entropy, lw=1, c='k')                               
        plt.scatter(x_normalized_iids,  y_distance_from_overall_entropy, marker='o', s=50, c='k')
        plt.xlabel('Minutes since peak')
        plt.ylabel('Distance from overall entropy')
        plt.grid(True)
        savefig(output_file_format%'distace_from_overall_entropy')
        plt.clf()   
        
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9)
        plt.subplot(111)
        plt.xlim(xmin=-20, xmax=120)
        plt.ylim(ymin=0.797, ymax=0.84)
        plt.plot(x_normalized_iids, y_focuses, lw=1, c='k')
        plt.scatter(x_normalized_iids, y_focuses, lw=1, marker='o', s=50, c='k')     
        plt.xlabel('Minutes since peak')
        plt.ylabel('Interval focus')
        plt.grid(True)
        savefig(output_file_format%'focus')
        plt.clf()
        
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9)
        plt.subplot(111)
        plt.xlim(xmin=-20, xmax=400)
#        plt.ylim(ymin=-0.43, ymax=-0.19)
        plt.plot(x_normalized_iids, y_distance_from_overall_focus, lw=1, c='k')                               
        plt.scatter(x_normalized_iids, y_distance_from_overall_focus, marker='o', s=50, c='k')   
        plt.xlabel('Minutes since peak')
        plt.ylabel('Distance from overall focus')
        plt.grid(True)
        savefig(output_file_format%'distace_from_overall_focus')

    @staticmethod
    def run():
#        DataAnalysis.hashtag_distribution_loglog()
        DataAnalysis.hashtag_locations_distribution_loglog()
#        DataAnalysis.fraction_of_occurrences_vs_rank_of_location()
#        DataAnalysis.top_k_locations_on_world_map()
#        DataAnalysis.content_affinity_vs_distance()
#        DataAnalysis.temporal_affinity_vs_distance()
#        DataAnalysis.spatial_metrics_cdf()
#        DataAnalysis.spatial_metrics_vs_occurrence_count()
#        DataAnalysis.ef_plot()
#        DataAnalysis.coverage_vs_spatial_properties()
#        DataAnalysis.peak_stats()
#        DataAnalysis.iid_vs_cumulative_distribution_and_peak_distribution()
#        DataAnalysis.ef_plots_for_peak()
#        DataAnalysis.norm_iid_vs_locality_measuers()

if __name__ == '__main__':
    DataAnalysis.run()