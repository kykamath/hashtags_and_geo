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
from library.mrjobwrapper import runMRJob
from library.plotting import savefig
from library.stats import filter_outliers
from mr_predict_hashtags_analysis import HashtagsExtractor
from mr_predict_hashtags_analysis import HashtagsWithMajorityInfo
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
f_impact_of_using_locations_to_predict = analysis_folder%'impact_of_using_locations_to_predict/'+\
                                                                                'impact_of_using_locations_to_predict'

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
    def impact_of_using_locations_to_predict():
        runMRJob(
                     ImpactOfUsingLocationsToPredict,
                     f_impact_of_using_locations_to_predict,
                     [df_hashtags_extractor],
                     jobconf={'mapred.reduce.tasks':100}
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
        MRAnalysis.impact_of_using_locations_to_predict()
        
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
        plt.ylabel('CCDF')
        savefig(output_file)
    @staticmethod
    def ccdf_time_at_which_hashtag_propagates_to_a_location():
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
        current_val=1.0
        for perct_of_utm_id in perct_of_utm_ids:
            perct_of_utm_ids1.append(current_val)
            current_val-=perct_of_utm_id
        perct_of_utm_ids = perct_of_utm_ids1
        ax = plt.subplot(111)
        ax.set_xscale('log')
        temp_map = dict(zip(majority_threshold_bucket_time, perct_of_utm_ids))
        print 'Percentage of locations propagated in first 6 hours: ', 1 - temp_map[360]
        print 'Percentage of locations between 1 and 6 hours: ', temp_map[60] - temp_map[360]
        plt.plot(majority_threshold_bucket_time, perct_of_utm_ids, c='k')
        plt.scatter(majority_threshold_bucket_time, perct_of_utm_ids, c='k')
        plt.grid(True)
        plt.xlabel('Time at which hashtag propagates to a location (minutes)')
        plt.ylabel('CCDF')
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
#        PredictHashtagsForLocationsPlots.perct_of_hashtag_occurrences_vs_time_of_propagation()
#        PredictHashtagsForLocationsPlots.ccdf_num_of_utmids_where_hashtag_propagates()
#        PredictHashtagsForLocationsPlots.ccdf_time_at_which_hashtag_propagates_to_a_location()
        PredictHashtagsForLocationsPlots.impact_of_using_location_to_predict_hashtag()
#        PredictHashtagsForLocationsPlots.example_of_hashtag_propagation_patterns()
        
if __name__ == '__main__':
    MRAnalysis.run()
#    PredictHashtagsForLocationsPlots.run()
