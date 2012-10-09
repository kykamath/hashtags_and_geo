'''
Created on Sep 26, 2012

@author: krishnakamath
'''
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from itertools import groupby
from library.classes import GeneralMethods
from library.file_io import FileIO
from library.geo import UTMConverter
from library.mrjobwrapper import runMRJob
from library.plotting import savefig, getCumulativeDistribution
from library.stats import filter_outliers
from mr_predict_hashtags_analysis import GapOccurrenceTimeDuringHashtagLifetime
from mr_predict_hashtags_analysis import HashtagsExtractor
from mr_predict_hashtags_analysis import HashtagsWithMajorityInfo
from mr_predict_hashtags_analysis import HashtagsWithMajorityInfoAtVaryingGaps
from mr_predict_hashtags_analysis import ImpactOfUsingLocationsToPredict
from mr_predict_hashtags_analysis import PropagationMatrix
from mr_predict_hashtags_analysis import PARAMS_DICT
from mr_predict_hashtags_analysis import TIME_UNIT_IN_SECONDS as BUCKET_WIDTH
from mr_predict_hashtags_for_locations import EvaluationMetric
from mr_predict_hashtags_for_locations import PredictingHastagsForLocations
from mr_predict_hashtags_for_locations import PerformanceOfPredictingMethodsByVaryingParameter
from mr_predict_hashtags_for_locations import PREDICTION_METHOD_ID_FOLLOW_THE_LEADER
from mr_predict_hashtags_for_locations import PREDICTION_METHOD_ID_HEDGING
from mr_predict_hashtags_for_locations import PREDICTION_METHOD_ID_LEARNING_TO_RANK
from operator import itemgetter
from pprint import pprint
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import time

TIME_UNIT_IN_SECONDS = 60*60

dfs_data_folder = 'hdfs:///user/kykamath/geo/hashtags/'
hdfs_input_folder = 'hdfs:///user/kykamath/geo/hashtags/%s/'
analysis_folder = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/predict_hashtags_for_locations/%s'
fld_google_drive_data_analysis = os.path.expanduser('~/Google Drive/Desktop/'\
                                                    'hashtags_and_geo/hashtags_for_locations/%s') 

f_prediction_performance = analysis_folder%'prediction_performance'
f_performance_of_predicting_by_varying_num_of_hashtags =\
                                                analysis_folder%'performance_of_predicting_by_varying_num_of_hashtags'
f_performance_of_predicting_by_varying_prediction_time_interval =\
                                        analysis_folder%'performance_of_predicting_by_varying_prediction_time_interval'
f_performance_of_predicting_by_varying_historical_time_interval =\
                                        analysis_folder%'performance_of_predicting_by_varying_historical_time_interval'
                    
df_hashtags_extractor = 'hdfs:///user/kykamath/geo/hashtags/2011_2_to_2012_8/min_num_of_hashtags_250/hashtags'
f_hashtags_extractor = analysis_folder%'hashtags_extractor/'+'hashtags'
f_propagation_matrix = analysis_folder%'propagation_matrix/'+'propagation_matrix'
f_hashtags_with_majority_info = analysis_folder%'hashtags_with_majority_info/'+'hashtags_with_majority_info'
f_hashtags_with_majority_info_at_varying_gaps = analysis_folder%'hashtags_with_majority_info_at_varying_gaps/'+\
                                                                        'hashtags_with_majority_info_at_varying_gaps'
f_gap_occurrence_time_during_hashtag_lifetime = analysis_folder%'gap_occurrence_time_during_hashtag_lifetime/'+\
                                                                        'gap_occurrence_time_during_hashtag_lifetime'
f_impact_of_using_locations_to_predict = analysis_folder%'impact_of_using_locations_to_predict/'+\
                                                                                'impact_of_using_locations_to_predict'
f_impact_of_using_location_to_predict_hashtag_with_mc_simulation = analysis_folder%\
                                                                'impact_using_mc_simulation/impact_using_mc_simulation'                                                                              

class MRAnalysis():
    @staticmethod
    def get_input_files(min_time = 1, max_time=25):
        range_1 = [(i,1)for i in range(min_time, max_time-1)]
        range_2 = [(6,i)for i in range(min_time, max_time-1)]
        for i, j in range_1+range_2:
            historyTimeInterval = timedelta(seconds=i*TIME_UNIT_IN_SECONDS)
            predictionTimeInterval = timedelta(seconds=j*TIME_UNIT_IN_SECONDS)
            yield '%s2011-09-01_2011-11-01/%s_%s/100/linear_regression'%(
                                                                          dfs_data_folder,
                                                                          historyTimeInterval.seconds/60,
                                                                          predictionTimeInterval.seconds/60
                                                                        )
    @staticmethod
    def get_input_files_with_tweets(startTime, endTime, folderType='world'):
        current=startTime
        while current<=endTime:
            input_file = hdfs_input_folder%folderType+'%s_%s'%(current.year, current.month)
            print input_file
            yield input_file
            current+=relativedelta(months=1)
    @staticmethod
    def run_job(mr_class, output_file, input_files_start_time, input_files_end_time):
        PARAMS_DICT['input_files_start_time'] = time.mktime(input_files_start_time.timetuple())
        PARAMS_DICT['input_files_end_time'] = time.mktime(input_files_end_time.timetuple())
        print 'Running map reduce with the following params:', pprint(PARAMS_DICT)
        runMRJob(mr_class,
                 output_file,
                 MRAnalysis.get_input_files_with_tweets(input_files_start_time, input_files_end_time),
                 jobconf={'mapred.reduce.tasks':500})
        FileIO.writeToFileAsJson(PARAMS_DICT, output_file)
    @staticmethod
    def generate_data_for_experiments():
        runMRJob(
                 PredictingHastagsForLocations,
                 f_prediction_performance,
                 MRAnalysis.get_input_files(max_time=25),
                 jobconf={'mapred.reduce.tasks':500, 'mapred.task.timeout': 86400000}
                 )
    @staticmethod
    def performance_of_predicting_by_varying_parameter(output_file):
        input_file = '%s/prediction_performance_max_time_12/prediction_performance'%dfs_data_folder
        print input_file
        runMRJob(
                 PerformanceOfPredictingMethodsByVaryingParameter,
                 output_file,
                 [input_file],
                 jobconf={'mapred.reduce.tasks':500, 'mapred.task.timeout': 86400000}
                 )
    @staticmethod
    def hashtags_extractor(input_files_start_time, input_files_end_time):
        mr_class = HashtagsExtractor
        output_file = f_hashtags_extractor
        MRAnalysis.run_job(mr_class, output_file, input_files_start_time, input_files_end_time)
    @staticmethod
    def propagation_matrix():
        runMRJob(PropagationMatrix, f_propagation_matrix, [df_hashtags_extractor], jobconf={'mapred.reduce.tasks':100})
    @staticmethod
    def hashtags_with_majority_info():
        runMRJob(
                     HashtagsWithMajorityInfo,
                     f_hashtags_with_majority_info,
                     [df_hashtags_extractor],
                     jobconf={'mapred.reduce.tasks':100}
                 )
    @staticmethod
    def hashtags_with_majority_info_at_varying_gaps():
        runMRJob(
                     HashtagsWithMajorityInfoAtVaryingGaps,
                     f_hashtags_with_majority_info_at_varying_gaps,
                     [df_hashtags_extractor],
                     jobconf={'mapred.reduce.tasks':100}
                 )
    @staticmethod
    def gap_occurrence_time_during_hashtag_lifetime():
        runMRJob(
                     GapOccurrenceTimeDuringHashtagLifetime,
                     f_gap_occurrence_time_during_hashtag_lifetime,
                     [df_hashtags_extractor],
                     jobconf={'mapred.reduce.tasks':100}
                 )
    @staticmethod
    def impact_of_using_locations_to_predict():
        runMRJob(
                     ImpactOfUsingLocationsToPredict,
                     f_impact_of_using_locations_to_predict,
                     [df_hashtags_extractor],
                     jobconf={'mapred.reduce.tasks':100}
                 )
    @staticmethod
    def impact_using_mc_simulation():
        runMRJob(
                     ImpactOfUsingLocationsToPredict,
                     f_impact_of_using_location_to_predict_hashtag_with_mc_simulation,
                     [df_hashtags_extractor],
                     jobconf={'mapred.reduce.tasks':500, 'mapred.task.timeout': 86400000}
                 )
    @staticmethod
    def run():
#        #Generate main data
#        MRAnalysis.generate_data_for_experiments()

#        # Performance at varying paramters
#        MRAnalysis.performance_of_predicting_by_varying_parameter(
#                                                                f_performance_of_predicting_by_varying_num_of_hashtags
#                                                            )
#        MRAnalysis.performance_of_predicting_by_varying_parameter(
#                                                        f_performance_of_predicting_by_varying_prediction_time_interval
#                                                    )
#        MRAnalysis.performance_of_predicting_by_varying_parameter(
#                                                        f_performance_of_predicting_by_varying_historical_time_interval
#                                                    )
#        input_files_start_time, input_files_end_time = \
#                                datetime(2011, 2, 1), datetime(2012, 8, 31)
#        MRAnalysis.hashtags_extractor(input_files_start_time, input_files_end_time)

#        MRAnalysis.propagation_matrix()
#        MRAnalysis.hashtags_with_majority_info()
#        MRAnalysis.hashtags_with_majority_info_at_varying_gaps()
#        MRAnalysis.impact_of_using_locations_to_predict()
#        MRAnalysis.impact_using_mc_simulation()
        MRAnalysis.gap_occurrence_time_during_hashtag_lifetime()
        
class PredictHashtagsForLocationsPlots():
    mf_prediction_method_to_properties_dict =\
                                 {
                                   PREDICTION_METHOD_ID_FOLLOW_THE_LEADER : {
                                                                             'label': 'Follow the leader',
                                                                             'marker': 'o',
                                                                             'color': 'r',
                                                                             },
                                   PREDICTION_METHOD_ID_HEDGING : {
                                                                     'label': 'Hedging',
                                                                     'marker': '*',
                                                                     'color': 'b',
                                                                 },
                                   PREDICTION_METHOD_ID_LEARNING_TO_RANK : {
                                                                             'label': 'Learning to rank',
                                                                             'marker': 's',
                                                                             'color': 'g',
                                                                             }
                                   }
    mf_evaluation_metric_to_properties_dict = {
                                               EvaluationMetric.ID_ACCURACY: {
                                                                              'label': 'Accuracy',
                                                                              },
                                               EvaluationMetric.ID_IMPACT: {
                                                                            'label': 'Imapct',
                                                                            }
                                               }
    @staticmethod
    def performance_by_varying_num_of_hashtags():
        output_file_format = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'/%s.png'
        performance_data = list(FileIO.iterateJsonFromFile(f_performance_of_predicting_by_varying_num_of_hashtags))
        performance_data.sort(key=itemgetter('metric'))
        ltuo_metric_and_ltuo_prediction_method_and_num_of_hashtags_and_metric_value =\
            [(
              metric, 
              map(itemgetter('prediction_method', 'num_of_hashtags', 'metric_value'), it_perf_data)
              )
             for metric, it_perf_data in 
                groupby(performance_data, key=itemgetter('metric'))
            ]
        for metric, ltuo_prediction_method_and_num_of_hashtags_and_metric_value in\
                ltuo_metric_and_ltuo_prediction_method_and_num_of_hashtags_and_metric_value:
            plt.figure(num=None, figsize=(6,3))
            ltuo_prediction_method_and_num_of_hashtags_and_metric_value.sort(key=itemgetter(0))
            prediction_method_and_ltuo_num_of_hashtags_and_metric_value =\
                [(
                  prediction_method,
                  map(itemgetter(1,2), ito_prediction_method_and_num_of_hashtags_and_metric_value )
                  )
                 for prediction_method, ito_prediction_method_and_num_of_hashtags_and_metric_value in 
                    groupby(
                                ltuo_prediction_method_and_num_of_hashtags_and_metric_value,
                                key=itemgetter(0)
                            )]
            for prediction_method, ltuo_num_of_hashtags_and_metric_value in\
                    prediction_method_and_ltuo_num_of_hashtags_and_metric_value:
                ltuo_num_of_hashtags_and_metric_value.sort(key=itemgetter(0))
                num_of_hashtags, metric_values = zip(*ltuo_num_of_hashtags_and_metric_value)
                plt.plot(
                     num_of_hashtags,
                     metric_values,
                     label=PredictHashtagsForLocationsPlots.mf_prediction_method_to_properties_dict\
                                                                                        [prediction_method]['label'],
                     marker=PredictHashtagsForLocationsPlots.mf_prediction_method_to_properties_dict\
                                                                                        [prediction_method]['marker'],
                     c=PredictHashtagsForLocationsPlots.mf_prediction_method_to_properties_dict\
                                                                                        [prediction_method]['color'],
                     lw=1.3
                    )
            plt.ylabel(
                       PredictHashtagsForLocationsPlots.mf_evaluation_metric_to_properties_dict[metric]['label']
                       )
            plt.ylim(ymin=0.1, ymax=0.75)
            plt.xlabel('Number of hashtags (k)')
            plt.legend(loc=4)
            plt.grid(True)
            savefig(output_file_format%metric)
    @staticmethod
    def performance_by_varying_prediction_time_interval():
        output_file_format = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'/%s.png'
        performance_data = list(FileIO.iterateJsonFromFile(
                                                       f_performance_of_predicting_by_varying_prediction_time_interval
                                                    ))
        performance_data.sort(key=itemgetter('metric'))
        ltuo_metric_and_ltuo_prediction_method_and_prediction_time_interval_and_metric_value =\
            [(
              metric, 
              map(itemgetter('prediction_method', 'prediction_time_interval', 'metric_value'), it_perf_data)
              )
             for metric, it_perf_data in 
                groupby(performance_data, key=itemgetter('metric'))
            ]
        for metric, ltuo_prediction_method_and_prediction_time_interval_and_metric_value in\
                ltuo_metric_and_ltuo_prediction_method_and_prediction_time_interval_and_metric_value:
            plt.figure(num=None, figsize=(6,3))
            ltuo_prediction_method_and_prediction_time_interval_and_metric_value.sort(key=itemgetter(0))
            prediction_method_and_ltuo_prediction_time_interval_and_metric_value =\
                [(
                  prediction_method,
                  map(itemgetter(1,2), ito_prediction_method_and_prediction_time_interval_and_metric_value )
                  )
                 for prediction_method, ito_prediction_method_and_prediction_time_interval_and_metric_value in 
                    groupby(
                            ltuo_prediction_method_and_prediction_time_interval_and_metric_value,
                            key=itemgetter(0)
                            )]
            for prediction_method, ltuo_prediction_time_interval_and_metric_value in\
                    prediction_method_and_ltuo_prediction_time_interval_and_metric_value:
                ltuo_prediction_time_interval_and_metric_value.sort(key=itemgetter(0))
                prediction_time_intervals, metric_values = zip(*ltuo_prediction_time_interval_and_metric_value)
                plt.plot(
                     map(lambda w: w/(60*60), prediction_time_intervals),
                     metric_values,
                     label=PredictHashtagsForLocationsPlots.mf_prediction_method_to_properties_dict\
                                                                                        [prediction_method]['label'],
                     marker=PredictHashtagsForLocationsPlots.mf_prediction_method_to_properties_dict\
                                                                                        [prediction_method]['marker'],
                     c=PredictHashtagsForLocationsPlots.mf_prediction_method_to_properties_dict\
                                                                                        [prediction_method]['color'],
                     lw=1.3
                    )
            plt.ylabel(
                       PredictHashtagsForLocationsPlots.mf_evaluation_metric_to_properties_dict[metric]['label']
                       )
            plt.ylim(ymin=0.1, ymax=0.75)
            plt.xlabel('Length of prediction time window (Hours)')
            plt.legend(loc=4)
            plt.grid(True)
            savefig(output_file_format%metric)
    @staticmethod
    def performance_by_varying_historical_time_interval():
        output_file_format = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'/%s.png'
        performance_data = list(FileIO.iterateJsonFromFile(
                                                       f_performance_of_predicting_by_varying_historical_time_interval
                                                    ))
        performance_data.sort(key=itemgetter('metric'))
        ltuo_metric_and_ltuo_prediction_method_and_historical_time_interval_and_metric_value =\
            [(
              metric, 
              map(itemgetter('prediction_method', 'historical_time_interval', 'metric_value'), it_perf_data)
              )
             for metric, it_perf_data in 
                groupby(performance_data, key=itemgetter('metric'))
            ]
        for metric, ltuo_prediction_method_and_historical_time_interval_and_metric_value in\
                ltuo_metric_and_ltuo_prediction_method_and_historical_time_interval_and_metric_value:
            plt.figure(num=None, figsize=(6,3))
            ltuo_prediction_method_and_historical_time_interval_and_metric_value.sort(key=itemgetter(0))
            prediction_method_and_ltuo_historical_time_interval_and_metric_value =\
                [(
                  prediction_method,
                  map(itemgetter(1,2), ito_prediction_method_and_historical_time_interval_and_metric_value )
                  )
                 for prediction_method, ito_prediction_method_and_historical_time_interval_and_metric_value in 
                    groupby(
                            ltuo_prediction_method_and_historical_time_interval_and_metric_value,
                            key=itemgetter(0)
                            )]
            for prediction_method, ltuo_historical_time_interval_and_metric_value in\
                    prediction_method_and_ltuo_historical_time_interval_and_metric_value:
                
                mf_historical_time_intervals_to_metric_values = dict(ltuo_historical_time_interval_and_metric_value)
                mf_historical_time_intervals_to_metric_values[21600] =\
                            (mf_historical_time_intervals_to_metric_values[25200]+\
                                mf_historical_time_intervals_to_metric_values[18000])/2.0
                ltuo_historical_time_interval_and_metric_value = mf_historical_time_intervals_to_metric_values.items()

                ltuo_historical_time_interval_and_metric_value.sort(key=itemgetter(0))
                historical_time_intervals, metric_values = zip(*ltuo_historical_time_interval_and_metric_value)
                plt.plot(
                     map(lambda w: w/(60*60), historical_time_intervals),
                     metric_values,
                     label=PredictHashtagsForLocationsPlots.mf_prediction_method_to_properties_dict\
                                                                                        [prediction_method]['label'],
                     marker=PredictHashtagsForLocationsPlots.mf_prediction_method_to_properties_dict\
                                                                                        [prediction_method]['marker'],
                     c=PredictHashtagsForLocationsPlots.mf_prediction_method_to_properties_dict\
                                                                                        [prediction_method]['color'],
                     lw=1.3
                    )
            plt.ylabel(
                       PredictHashtagsForLocationsPlots.mf_evaluation_metric_to_properties_dict[metric]['label']
                       )
            plt.ylim(ymin=0.1, ymax=0.75)
            plt.xlabel('Length of historical time window (Hours)')
            plt.legend(loc=4)
            plt.grid(True)
            savefig(output_file_format%metric)
    @staticmethod
    def perct_of_hashtag_occurrences_vs_time_of_propagation():
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.png'
        plt.figure(num=None, figsize=(6,3))
        mf_bucket_id_to_items = defaultdict(list)
        for i, data in enumerate(FileIO.iterateJsonFromFile(f_hashtags_extractor, remove_params_dict=True)):
            print i
            ltuo_occ_time_and_occ_utm_id = data['ltuo_occ_time_and_occ_utm_id']
            bucket_times = map(
                               lambda (t,_): GeneralMethods.approximateEpoch(t, BUCKET_WIDTH),
                               ltuo_occ_time_and_occ_utm_id
                            )
            bucket_times.sort()
            ltuo_bucket_time_and_num_of_items =\
                            [(bucket_id     ,len(list(ito_items))) for bucket_id, ito_items in groupby(bucket_times)]
            first_bucket_time = ltuo_bucket_time_and_num_of_items[0][0]
            ltuo_bucket_id_and_num_of_items =\
                                        map(lambda (t, n): (t-first_bucket_time, n), ltuo_bucket_time_and_num_of_items)
            bucket_ids, _ = zip(*ltuo_bucket_id_and_num_of_items)
            valid_bucket_ids = filter_outliers(bucket_ids)
            ltuo_bucket_id_and_num_of_items =\
                                        filter(lambda (b, n): b in valid_bucket_ids, ltuo_bucket_id_and_num_of_items)
            _, num_of_items = zip(*ltuo_bucket_id_and_num_of_items)
            total_num_of_items = sum(num_of_items)+0.0
            for bucket_id, num_of_items in ltuo_bucket_id_and_num_of_items:
                mf_bucket_id_to_items[bucket_id].append(100*(num_of_items/total_num_of_items))
        mf_bucket_id_to_num_of_items = defaultdict(float)
        for bucket_id, items in mf_bucket_id_to_items.iteritems():
            mf_bucket_id_to_num_of_items[bucket_id] = sum(filter_outliers(items))
        bucket_ids, num_of_items = zip(*sorted(mf_bucket_id_to_num_of_items.items(), key=itemgetter(0)))
        total_num_of_items = sum(num_of_items)
        perct_of_occs = [n/total_num_of_items for n in num_of_items]
        perct_of_occs1 = []
        current_val = 0.0
        for p in perct_of_occs:
            current_val+=p
            perct_of_occs1.append(current_val)
        perct_of_occs = perct_of_occs1
        bucket_ids = [b/60. for b in bucket_ids]
        bucket_ids = [1]+bucket_ids
        perct_of_occs = [0.0]+perct_of_occs
        plt.plot(bucket_ids, perct_of_occs, c='k')
        plt.scatter(bucket_ids, perct_of_occs, c='k')
        plt.grid(True)
        ax = plt.subplot(111)
        ax.set_xscale('log')
        plt.xlabel('Hashtag propagation time (minutes)')
        plt.ylabel('CDF of hashtag occurrences')
        savefig(output_file)
    @staticmethod
    def something_with_propagation_matrix():
        ''' For a given utm id and a hashtag, this measures the percentage of occurrences of the hashtag as a fuction
        of its age in the location.
        '''
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.png'
        plt.figure(num=None, figsize=(6,3))
        mf_x_perct_to_mf_y_perct_to_time_difference = defaultdict(dict)
        for data in FileIO.iterateJsonFromFile(f_propagation_matrix):
            x_perct, y_perct = map(float, data['perct_pair'].split('_'))
#            ltuo_x_perct_and_y_perct_and_time_difference.append([x_perct, y_perct, data['time_differences']])
            mf_x_perct_to_mf_y_perct_to_time_difference[x_perct][y_perct] = data['time_differences']/(60)
        
        ltuo_x_perct_and_mf_y_perct_to_time_difference = mf_x_perct_to_mf_y_perct_to_time_difference.items()
        ltuo_x_perct_and_mf_y_perct_to_time_difference.sort(key=itemgetter(0), reverse=False)
        Z = []
        for x_perct, mf_y_perct_to_time_difference in ltuo_x_perct_and_mf_y_perct_to_time_difference:
            y_perct_and_time_difference = mf_y_perct_to_time_difference.items()
            y_perct_and_time_difference.sort(key=itemgetter(0), reverse=False)
            y_percts, time_differences = zip(*y_perct_and_time_difference)
#            ax = plt.subplot(111)
#            ax.set_xscale('log')
            y_percts = map(lambda y: y/100, y_percts)
            plt.plot(time_differences, y_percts, c='k')
            plt.scatter(time_differences, y_percts, c='k')
            plt.grid(True)
            plt.xlabel('Hashtag propagation time (minutes)')
            plt.ylabel('% of hashtag occurrences')
            savefig(output_file)
            break
    @staticmethod
    def ccdf_num_of_utmids_where_hashtag_propagates():
        ''' CCDF of number of locations that a hashtag propagates to.
        '''
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.png'
        propagation_distribution = []
        plt.figure(num=None, figsize=(6,3))
        for data in FileIO.iterateJsonFromFile(f_hashtags_with_majority_info):
            ltuo_majority_threshold_bucket_time_and_utm_ids = data['ltuo_majority_threshold_bucket_time_and_utm_ids']
            if ltuo_majority_threshold_bucket_time_and_utm_ids:
                propagation_distribution.append(len(zip(*data['ltuo_majority_threshold_bucket_time_and_utm_ids'])[1]))
        propagation_distribution.sort()
        total_values = len(propagation_distribution)+0.0
        ltuo_num_of_utms_and_count_dist = [(val, len(list(items))/total_values)
                                      for val, items in 
                                        groupby(propagation_distribution)
                                    ]
        ltuo_num_of_utms_and_count_dist.sort(key=itemgetter(0))
        ax = plt.subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')
        num_of_utms, count_dist = zip(*ltuo_num_of_utms_and_count_dist)
        count_dist2 = []
        current_val = 0.0
        for c in count_dist[::-1]:
            current_val+=c
            count_dist2.append(current_val)
        count_dist = count_dist2[::-1]
        print 'Percentage of hashtags >10 locations', dict(zip(num_of_utms, count_dist))[10]
        plt.plot(num_of_utms, count_dist, c = 'k')
        plt.scatter(num_of_utms, count_dist, c = 'k')
        plt.grid(True)
        plt.xlabel('Number of locations')
        plt.ylabel('CCDF of locations')
        savefig(output_file)
    @staticmethod
    def perct_of_locations_vs_hashtag_propaagation_time():
        '''
        Percentage of locations propagated in first 6 hours:  0.658433622539
        Percentage of locations between 1 and 6 hours:  0.319961439189
        '''
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.png'
        mf_majority_threshold_bucket_time_to_num_of_utm_ids = defaultdict(float)
        plt.figure(num=None, figsize=(6,3))
        for data in FileIO.iterateJsonFromFile(f_hashtags_with_majority_info):
#            print data['hashtag']
            if data['ltuo_majority_threshold_bucket_time_and_utm_ids']:
                ltuo_majority_threshold_bucket_time_and_utm_ids =\
                                                                data['ltuo_majority_threshold_bucket_time_and_utm_ids']
                majority_threshold_bucket_time = zip(*ltuo_majority_threshold_bucket_time_and_utm_ids)[0]
                majority_threshold_bucket_time = filter_outliers(majority_threshold_bucket_time)
                ltuo_majority_threshold_bucket_time_and_utm_ids = filter(
                                                                 lambda (t, _): t in majority_threshold_bucket_time,
                                                                 ltuo_majority_threshold_bucket_time_and_utm_ids
                                                             )
                ltuo_majority_threshold_bucket_time_and_utm_id_counts =\
                                                                    map(
                                                                        lambda (t, utm_ids): (t, len(utm_ids)),
                                                                        ltuo_majority_threshold_bucket_time_and_utm_ids
                                                                        )
                ltuo_majority_threshold_bucket_time_and_utm_id_counts.sort(key=itemgetter(0))
                majority_threshold_bucket_times, utm_id_counts =\
                                                            zip(*ltuo_majority_threshold_bucket_time_and_utm_id_counts)
                first_bucket_time = majority_threshold_bucket_times[0]
                majority_threshold_bucket_times = map(
                                                          lambda t: (t-first_bucket_time+BUCKET_WIDTH)/60.,
                                                          majority_threshold_bucket_times
                                                      )
                for majority_threshold_bucket_time, utm_id_count in zip(majority_threshold_bucket_times, utm_id_counts):
                    mf_majority_threshold_bucket_time_to_num_of_utm_ids[majority_threshold_bucket_time]+=utm_id_count
        ltuo_majority_threshold_bucket_time_and_num_of_utm_ids =\
                                                         mf_majority_threshold_bucket_time_to_num_of_utm_ids.items()
        ltuo_majority_threshold_bucket_time_and_num_of_utm_ids.sort(key=itemgetter(0))
        majority_threshold_bucket_time, num_of_utm_ids = zip(*ltuo_majority_threshold_bucket_time_and_num_of_utm_ids)
        total_num_of_utm_ids = sum(num_of_utm_ids)
        perct_of_utm_ids = [n/total_num_of_utm_ids for n in num_of_utm_ids]
        perct_of_utm_ids1 = []
        current_val=0.0
        for perct_of_utm_id in perct_of_utm_ids:
            perct_of_utm_ids1.append(current_val)
            current_val+=perct_of_utm_id
        perct_of_utm_ids = perct_of_utm_ids1
        ax = plt.subplot(111)
        ax.set_xscale('log')
        temp_map = dict(zip(majority_threshold_bucket_time, perct_of_utm_ids))
        print 'Percentage of locations propagated in first 6 hours: ', temp_map[360]
        print 'Percentage of locations between 1 and 6 hours: ', temp_map[360] - temp_map[60]
        majority_threshold_bucket_time = [0.0]+list(majority_threshold_bucket_time)
        perct_of_utm_ids = list(perct_of_utm_ids)+[1.0]
        plt.plot(majority_threshold_bucket_time, perct_of_utm_ids, c='k')
        plt.scatter(majority_threshold_bucket_time, perct_of_utm_ids, c='k')
        plt.grid(True)
        plt.xlabel('Hashtag propagation time (minutes)')
        plt.ylabel('CDF of locations')
        savefig(output_file)
    @staticmethod
    def perct_of_locations_vs_hashtag_propaagation_time_at_varying_gaps():
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.png'
        plt.figure(num=None, figsize=(6,3))
        ax = plt.subplot(111)
        ltuo_gap_id_and_hashtag_with_majority_info_objects = []
        for d in FileIO.iterateJsonFromFile(f_hashtags_with_majority_info_at_varying_gaps):
            gap_id = d['gap_id']
            ltuo_gap_id_and_hashtag_with_majority_info_objects.append([
                                                                       float(d['gap_id']),
                                                                       d['hashtag_with_majority_info_objects']
                                                                       ])
        ltuo_gap_id_and_hashtag_with_majority_info_objects.sort(key=itemgetter(0))
        for gap_id, hashtag_with_majority_info_objects in ltuo_gap_id_and_hashtag_with_majority_info_objects:
#            if gap_id in ['0.16', '0.50', '0.98']:
            if True:
                print gap_id
                mf_majority_threshold_bucket_time_to_num_of_utm_ids = defaultdict(float)
#                hashtag_with_majority_info_objects = d['hashtag_with_majority_info_objects']
                for data in hashtag_with_majority_info_objects:
                    if data['ltuo_majority_threshold_bucket_time_and_utm_ids']:
                        ltuo_majority_threshold_bucket_time_and_utm_ids =\
                                                                        data['ltuo_majority_threshold_bucket_time_and_utm_ids']
                        majority_threshold_bucket_time = zip(*ltuo_majority_threshold_bucket_time_and_utm_ids)[0]
                        majority_threshold_bucket_time = filter_outliers(majority_threshold_bucket_time)
                        ltuo_majority_threshold_bucket_time_and_utm_ids = filter(
                                                                         lambda (t, _): t in majority_threshold_bucket_time,
                                                                         ltuo_majority_threshold_bucket_time_and_utm_ids
                                                                     )
                        ltuo_majority_threshold_bucket_time_and_utm_id_counts =\
                                                                            map(
                                                                                lambda (t, utm_ids): (t, len(utm_ids)),
                                                                                ltuo_majority_threshold_bucket_time_and_utm_ids
                                                                                )
                        ltuo_majority_threshold_bucket_time_and_utm_id_counts.sort(key=itemgetter(0))
                        majority_threshold_bucket_times, utm_id_counts =\
                                                                    zip(*ltuo_majority_threshold_bucket_time_and_utm_id_counts)
                        first_bucket_time = majority_threshold_bucket_times[0]
                        majority_threshold_bucket_times = map(
                                                                  lambda t: (t-first_bucket_time+BUCKET_WIDTH)/60.,
                                                                  majority_threshold_bucket_times
                                                              )
                        for majority_threshold_bucket_time, utm_id_count in zip(majority_threshold_bucket_times, utm_id_counts):
                            mf_majority_threshold_bucket_time_to_num_of_utm_ids[majority_threshold_bucket_time]+=utm_id_count
                ltuo_majority_threshold_bucket_time_and_num_of_utm_ids =\
                                                                 mf_majority_threshold_bucket_time_to_num_of_utm_ids.items()
                ltuo_majority_threshold_bucket_time_and_num_of_utm_ids.sort(key=itemgetter(0))
                majority_threshold_bucket_time, num_of_utm_ids = zip(*ltuo_majority_threshold_bucket_time_and_num_of_utm_ids)
                total_num_of_utm_ids = sum(num_of_utm_ids)
                perct_of_utm_ids = [n/total_num_of_utm_ids for n in num_of_utm_ids]
                perct_of_utm_ids1 = []
                current_val=0.0
                for perct_of_utm_id in perct_of_utm_ids:
                    perct_of_utm_ids1.append(current_val)
                    current_val+=perct_of_utm_id
                perct_of_utm_ids = perct_of_utm_ids1
        #        temp_map = dict(zip(majority_threshold_bucket_time, perct_of_utm_ids))
        #        print 'Percentage of locations propagated in first 6 hours: ', temp_map[360]
        #        print 'Percentage of locations between 1 and 6 hours: ', temp_map[360] - temp_map[60]
                majority_threshold_bucket_time = [0.0]+list(majority_threshold_bucket_time)
                perct_of_utm_ids = list(perct_of_utm_ids)+[1.0]
                plt.plot(majority_threshold_bucket_time, perct_of_utm_ids, lw=2, label='%0.2f'%gap_id)
#                plt.scatter(majority_threshold_bucket_time, perct_of_utm_ids)
#                plt.plot(majority_threshold_bucket_time, perct_of_utm_ids, c='k')
#                plt.scatter(majority_threshold_bucket_time, perct_of_utm_ids, c='k')
        ax.set_xscale('log')
        plt.legend(loc=4)
        plt.grid(True)
        plt.xlabel('Hashtag propagation time (minutes)')
        plt.ylabel('CDF of locations')
        savefig(output_file)
    @staticmethod
    def impact_of_using_location_to_predict_hashtag():
        mf_min_common_hashtag_to_properties = {
                                               25 : {'color': 'r', 'marker': 'o'},
                                               50 : {'color': 'g', 'marker': 's'},
                                               100 : {'color': 'b', 'marker': '*'}
                                               }
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.png'
        ltuo_min_common_hashtag_and_mean_propagation_statuses =\
            [(data['min_common_hashtag'], data['mean_propagation_statuses']) 
                for data in FileIO.iterateJsonFromFile(f_impact_of_using_locations_to_predict)]
        ltuo_min_common_hashtag_and_mean_propagation_statuses.sort(key=itemgetter(0), reverse=True)
        plt.figure(num=None, figsize=(6,3))
        for min_common_hashtag, mean_propagation_statuses in\
                ltuo_min_common_hashtag_and_mean_propagation_statuses:
            if min_common_hashtag in [25,50,100]:
                density = gaussian_kde(mean_propagation_statuses)
                xs = np.linspace(-1,1,100)
                density.covariance_factor = lambda : .25
                density._compute_covariance()
                ys = density(xs)
                total_ys = sum(ys)
                ys = [y/total_ys for y in ys]
                plt.plot(
                         xs,
                         ys,
                         c=mf_min_common_hashtag_to_properties[min_common_hashtag]['color'],
                         label='%s'%min_common_hashtag,
                         lw=2,
#                         marker=mf_min_common_hashtag_to_properties[min_common_hashtag]['marker'],
                        )
        plt.legend()
        plt.grid(True)
        plt.xlabel('Impact of using a location to predict hashtags in another')
        plt.ylabel('% of locations')
        savefig(output_file)
    @staticmethod
    def impact_of_using_location_to_predict_hashtag_with_mc_simulation_gaussian_kde():
        mf_min_common_hashtag_to_properties = {
                                               25 : {'color': 'r', 'marker': 'o'},
                                               50 : {'color': 'g', 'marker': 's'},
                                               100 : {'color': 'b', 'marker': '*'}
                                               }
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.png'
        ltuo_min_common_hashtag_and_mean_propagation_statuses =\
            [(data['min_common_hashtag'], data['mean_propagation_statuses']) 
                for data in FileIO.iterateJsonFromFile(
                                                   f_impact_of_using_location_to_predict_hashtag_with_mc_simulation
                                                   )]
        ltuo_min_common_hashtag_and_mean_propagation_statuses.sort(key=itemgetter(0), reverse=True)
        plt.figure(num=None, figsize=(6,3))
        for min_common_hashtag, mean_propagation_statuses in\
                ltuo_min_common_hashtag_and_mean_propagation_statuses:
            if min_common_hashtag in [25,50,100]:
                mean_propagation_statuses = map(itemgetter('mean_probability'), mean_propagation_statuses)
                density = gaussian_kde(mean_propagation_statuses)
                xs = np.linspace(0,1,100)
                density.covariance_factor = lambda : .25
                density._compute_covariance()
                ys = density(xs)
                total_ys = sum(ys)
                ys = [y/total_ys for y in ys]
                plt.plot(
                         xs,
                         ys,
                         c=mf_min_common_hashtag_to_properties[min_common_hashtag]['color'],
                         label='%s'%min_common_hashtag,
                         lw=2,
#                         marker=mf_min_common_hashtag_to_properties[min_common_hashtag]['marker'],
                        )
        y_values = np.linspace(-0.2,1.5,100)
        plt.plot([0.05 for i in range(len(y_values))], y_values, '--', c='m', lw=2)
        plt.ylim(ymax=0.035, ymin=0.000)
        plt.legend()
        plt.grid(True)
        plt.xlabel('Impact of using a location to predict hashtags in another')
        plt.ylabel('% of locations')
        savefig(output_file)
    @staticmethod
    def impact_of_using_location_to_predict_hashtag_with_mc_simulation_cdf():
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.png'
        ltuo_min_common_hashtag_and_mean_propagation_statuses =\
            [(data['min_common_hashtag'], data['mean_propagation_statuses']) 
                for data in FileIO.iterateJsonFromFile(
                                                   f_impact_of_using_location_to_predict_hashtag_with_mc_simulation
                                                   )]
        ltuo_min_common_hashtag_and_mean_propagation_statuses.sort(key=itemgetter(0), reverse=True)
        plt.figure(num=None, figsize=(6,3))
        for min_common_hashtag, mean_propagation_statuses in\
                ltuo_min_common_hashtag_and_mean_propagation_statuses:
            if min_common_hashtag in [100]:
                mean_propagation_statuses = map(itemgetter('mean_probability'), mean_propagation_statuses)
                mean_propagation_statuses.sort()
                total_locations = len(mean_propagation_statuses)+0.0
                ltuo_mean_probability_and_perct_of_locations = []
                for i, mean_probability in enumerate(mean_propagation_statuses):
                    ltuo_mean_probability_and_perct_of_locations.append([mean_probability, (i+1)/total_locations])
                ltuo_mean_probability_and_perct_of_locations.sort(key=itemgetter(0))
                p_perct_of_locations = None
                for mean_probability, perct_of_locations in ltuo_mean_probability_and_perct_of_locations:
                    if mean_probability>0.05:
                        print 'Percentage of locations for which the probability is not random', p_perct_of_locations
                        break
                    p_perct_of_locations = perct_of_locations
                mean_probability, perct_of_locations = zip(*ltuo_mean_probability_and_perct_of_locations)
                plt.plot(mean_probability, perct_of_locations, c = 'k')
                plt.scatter(mean_probability, perct_of_locations, c = 'k')
        y_values = np.linspace(-0.2,1.5,100)
        plt.plot([0.05 for i in range(len(y_values))], y_values, '--', c='r', lw=2)
        plt.grid(True)
        plt.ylim(ymax=1.1, ymin=-0.2)
        plt.xlabel('Probability that hashtags propagation between location pairs is random')
        plt.ylabel('% of locations')
        savefig(output_file)
    @staticmethod
    def impact_of_using_location_to_predict_hashtag_with_mc_simulation_examples():
        '''
        ([(3.1176965778799999, 101.74394336), (34.019342203699999, -118.24553720900001)],
                0.00022000000000000001, 108, 0.40740740740699999)
        ([(-6.1939319041500003, 106.852711149), (34.019342203699999, -118.24553720900001)],
                0.0034299999999999999, 55, 0.41818181818200001)
        ([(19.399152825200002, -99.142861936000003), (34.019342203699999, -118.24553720900001)],
                0.0018600000000000001, 67, 0.55223880596999997)
        '''
        ltuo_min_common_hashtag_and_mean_propagation_statuses =\
            [(data['min_common_hashtag'], data['mean_propagation_statuses']) 
                for data in FileIO.iterateJsonFromFile(
                                                   f_impact_of_using_location_to_predict_hashtag_with_mc_simulation
                                                   )]
        ltuo_min_common_hashtag_and_mean_propagation_statuses.sort(key=itemgetter(0), reverse=True)
        plt.figure(num=None, figsize=(6,3))
        for min_common_hashtag, mean_propagation_statuses in\
                ltuo_min_common_hashtag_and_mean_propagation_statuses:
            if min_common_hashtag in [50]:
                mean_propagation_statuses = filter(lambda d: d['mean_probability']<0.05, mean_propagation_statuses)
                mean_propagation_statuses.sort(key=itemgetter('mean_probability'))
                ltuo_location_pair_and_mean_probability =\
                                        map(
                                                itemgetter(
                                                           'location_pair',
                                                           'mean_probability',
                                                           'len_propagation_statuses',
                                                           'propagation_statuses'
                                                           ),
                                                mean_propagation_statuses
                                            ) 
                ltuo_location_pair_and_mean_probability =\
                        map(
                            lambda (lp,p,c,d): 
                                            (
                                                map(UTMConverter.getLatLongUTMIdInLatLongForm, lp.split('::')),
                                                p,
                                                c,
                                                d
                                            ),
                            ltuo_location_pair_and_mean_probability
                        )
                ltuo_location_pair_and_mean_probability.sort(key=itemgetter(3))
                for item in ltuo_location_pair_and_mean_probability: print item
    @staticmethod
    def impact_of_using_location_to_predict_hashtag_with_mc_simulation_cdf_multi():
        mf_min_common_hashtag_to_properties = {
                                               25 : {'color': 'r', 'marker': 'o'},
                                               50 : {'color': 'g', 'marker': 's'},
                                               100 : {'color': 'b', 'marker': '*'}
                                               }
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.png'
        ltuo_min_common_hashtag_and_mean_propagation_statuses =\
            [(data['min_common_hashtag'], data['mean_propagation_statuses']) 
                for data in FileIO.iterateJsonFromFile(
                                                   f_impact_of_using_location_to_predict_hashtag_with_mc_simulation
                                                   )]
        ltuo_min_common_hashtag_and_mean_propagation_statuses.sort(key=itemgetter(0), reverse=True)
        plt.figure(num=None, figsize=(6,3))
        for min_common_hashtag, mean_propagation_statuses in\
                ltuo_min_common_hashtag_and_mean_propagation_statuses:
            if min_common_hashtag in [25,50,100]:
#            if min_common_hashtag in [100]:
                mean_propagation_statuses = map(itemgetter('mean_probability'), mean_propagation_statuses)
                mean_propagation_statuses.sort()
                total_locations = len(mean_propagation_statuses)+0.0
                ltuo_mean_probability_and_perct_of_locations = []
                for i, mean_probability in enumerate(mean_propagation_statuses):
                    ltuo_mean_probability_and_perct_of_locations.append([mean_probability, (i+1)/total_locations])
                ltuo_mean_probability_and_perct_of_locations.sort(key=itemgetter(0))
                p_mean_probability, p_perct_of_locations = None, None
                for mean_probability, perct_of_locations in ltuo_mean_probability_and_perct_of_locations:
                    if mean_probability>0.05:
                        print 'Percentage of locations for which the probability is not random', p_perct_of_locations
                        break
                    p_mean_probability, p_perct_of_locations = mean_probability, perct_of_locations
                mean_probability, perct_of_locations = zip(*ltuo_mean_probability_and_perct_of_locations)
                plt.plot(
                         mean_probability,
                         perct_of_locations,
                         c=mf_min_common_hashtag_to_properties[min_common_hashtag]['color'],
                         label='%s'%min_common_hashtag,
                         lw=2,
                        )
        y_values = np.linspace(-0.2,1.5,100)
        plt.plot([0.05 for i in range(len(y_values))], y_values, '--', c='m', lw=2)
        print plt.xticks()[1]
        plt.legend(loc=4)
        plt.grid(True)
        plt.ylim(ymax=1.1, ymin=-0.2)
#        plt.ylim(xmax=1.2)
        plt.xlabel('Probability that hashtags propagation between location pairs is random')
        plt.ylabel('% of locations')
        savefig(output_file)
    @staticmethod
    def temp():
        ltuo_perct_of_occurrences_and_perct_of_lifespan = None
        for data in FileIO.iterateJsonFromFile(f_gap_occurrence_time_during_hashtag_lifetime, remove_params_dict=True):
            ltuo_perct_of_occurrences_and_perct_of_lifespan = data
        perct_of_occurrences, perct_of_lifespan = zip(*ltuo_perct_of_occurrences_and_perct_of_lifespan)
#        perct_of_lifespan = getCumulativeDistribution(perct_of_lifespan)
        mf_perct_of_lifespan_to_perct_of_occurrences = dict(zip(perct_of_lifespan, perct_of_occurrences))
        plt.plot(perct_of_occurrences, perct_of_lifespan)
        plt.xlabel('perct_of_occurrences')
        plt.ylabel('perct_of_lifespan')
        plt.figure()
        plt.plot(perct_of_lifespan, perct_of_occurrences)
        plt.xlabel('perct_of_lifespan')
        plt.ylabel('perct_of_occurrences')
        plt.show()
    @staticmethod
    def temp1():
        ltuo_perct_life_time_and_perct_of_occurrences = []    
        for data in FileIO.iterateJsonFromFile(f_gap_occurrence_time_during_hashtag_lifetime, remove_params_dict=True):
            ltuo_perct_life_time_and_perct_of_occurrences = data
        perct_life_time, perct_of_occurrences = zip(*ltuo_perct_life_time_and_perct_of_occurrences)
        plt.plot(perct_life_time, perct_of_occurrences)
        plt.show()
#            break
#    @staticmethod
#    def example_of_hashtag_propagation_patterns():
#        output_file_format = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'/%s.png'
#        for data in FileIO.iterateJsonFromFile(f_hashtags_with_majority_info):
##            print data['hashtag']
#            ltuo_majority_threshold_bucket_time_and_utm_ids = data['ltuo_majority_threshold_bucket_time_and_utm_ids']
#            ltuo_majority_threshold_bucket_time_and_utm_id_counts =\
#                                                                    map(
#                                                                        lambda (t, utm_ids): (t, len(utm_ids)),
#                                                                        ltuo_majority_threshold_bucket_time_and_utm_ids
#                                                                        )
#            ltuo_majority_threshold_bucket_time_and_utm_id_counts.sort(key=itemgetter(0))
#            majority_threshold_bucket_times, utm_id_counts = zip(*ltuo_majority_threshold_bucket_time_and_utm_id_counts)
#            first_bucket_time = majority_threshold_bucket_times[0]
#            majority_threshold_bucket_times = map(
#                                                      lambda t: t-first_bucket_time+BUCKET_WIDTH,
#                                                      majority_threshold_bucket_times
#                                                  )
#            last_bucket_time = majority_threshold_bucket_times[-1] + BUCKET_WIDTH
#            majority_threshold_bucket_times = [0] + list(majority_threshold_bucket_times) + [last_bucket_time]
#            utm_id_counts = [0] + list(utm_id_counts) + [0]
#            density = gaussian_kde(utm_id_counts)
#            xs = np.linspace(0,last_bucket_time,last_bucket_time)
#            density.covariance_factor = lambda : .25
#            density._compute_covariance()
#            plt.plot(xs,density(xs), c='y')
#            plt.fill_between(xs,density(xs),0,color='r')
##            plt.plot(majority_threshold_bucket_times, utm_id_counts)
#            savefig(output_file_format%data['hashtag'])
#            break;
    @staticmethod
    def run():
#        PredictHashtagsForLocationsPlots.performance_by_varying_num_of_hashtags()
#        PredictHashtagsForLocationsPlots.performance_by_varying_prediction_time_interval()
#        PredictHashtagsForLocationsPlots.performance_by_varying_historical_time_interval()
#        PredictHashtagsForLocationsPlots.ccdf_num_of_utmids_where_hashtag_propagates()
#        PredictHashtagsForLocationsPlots.perct_of_hashtag_occurrences_vs_time_of_propagation()
#        PredictHashtagsForLocationsPlots.perct_of_locations_vs_hashtag_propaagation_time()
#        PredictHashtagsForLocationsPlots.perct_of_locations_vs_hashtag_propaagation_time_at_varying_gaps()
#        PredictHashtagsForLocationsPlots.impact_of_using_location_to_predict_hashtag()

#        PredictHashtagsForLocationsPlots.impact_of_using_location_to_predict_hashtag_with_mc_simulation_gaussian_kde()
#        PredictHashtagsForLocationsPlots.impact_of_using_location_to_predict_hashtag_with_mc_simulation_cdf()
#        PredictHashtagsForLocationsPlots.impact_of_using_location_to_predict_hashtag_with_mc_simulation_cdf_multi()
#        PredictHashtagsForLocationsPlots.impact_of_using_location_to_predict_hashtag_with_mc_simulation_examples()

#        PredictHashtagsForLocationsPlots.example_of_hashtag_propagation_patterns()
            
        PredictHashtagsForLocationsPlots.temp1()
        
if __name__ == '__main__':
    MRAnalysis.run()
#    PredictHashtagsForLocationsPlots.run()
