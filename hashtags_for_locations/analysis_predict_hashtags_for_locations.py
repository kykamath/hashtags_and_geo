'''
Created on Sep 26, 2012

@author: krishnakamath
'''
import sys
sys.path.append('../')
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from itertools import chain, groupby
from hashtags_for_locations.models import ModelSelectionHistory
from hashtags_for_locations.models import PREDICTION_MODELS_PROPERTIES
from hashtags_for_locations.plots import MAP_FROM_MODEL_TO_COLOR
from hashtags_for_locations.plots import MAP_FROM_MODEL_TO_MARKER
from hashtags_for_locations.plots import MAP_FROM_MODEL_TO_MODEL_TYPE
from hashtags_for_locations.plots import LearningAnalysis
from library.classes import GeneralMethods
from library.file_io import FileIO
from library.geo import getLocationFromLid
from library.geo import isWithinBoundingBox
from library.geo import UTMConverter, plot_graph_clusters_on_world_map, plotPointsOnWorldMap
from library.geo import UTMConverter, plot_graph_clusters_on_world_map, plotPointsOnWorldMap
from library.geo import UTMConverter, plot_graph_clusters_on_world_map, plotPointsOnWorldMap
from library.graphs import clusterUsingAffinityPropagation, clusterUsingMCLClustering
from library.mrjobwrapper import runMRJob
from library.plotting import CurveFit
from library.plotting import getCumulativeDistribution
from library.plotting import getInverseCumulativeDistribution
from library.plotting import savefig
from library.plotting import splineSmooth
from library.stats import filter_outliers
from library.stats import getOutliersRangeUsingIRQ
from mr_predict_hashtags_analysis import GapOccurrenceTimeDuringHashtagLifetime
from mr_predict_hashtags_analysis import HashtagsExtractor
from mr_predict_hashtags_analysis import HashtagsWithMajorityInfo
from mr_predict_hashtags_analysis import HashtagsWithMajorityInfoAtVaryingGaps
from mr_predict_hashtags_analysis import ImpactOfUsingLocationsToPredict
from mr_predict_hashtags_analysis import LocationClusters
from mr_predict_hashtags_analysis import PropagationMatrix
from mr_predict_hashtags_analysis import PARAMS_DICT
from mr_predict_hashtags_analysis import TIME_UNIT_IN_SECONDS as BUCKET_WIDTH
from mr_predict_hashtags_for_locations import EvaluationMetric
from mr_predict_hashtags_for_locations import PredictingHastagsForLocations
from mr_predict_hashtags_for_locations import PerformanceByLocation
from mr_predict_hashtags_for_locations import PerformanceOfPredictingMethodsByVaryingParameter
from mr_predict_hashtags_for_locations import PREDICTION_METHOD_ID_FOLLOW_THE_LEADER
from mr_predict_hashtags_for_locations import PREDICTION_METHOD_ID_HEDGING
from mr_predict_hashtags_for_locations import PREDICTION_METHOD_ID_LEARNING_TO_RANK
import scipy
from operator import itemgetter
from pprint import pprint
from scipy.stats import gaussian_kde
from settings import timeUnitWithOccurrencesFile
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import time

TIME_UNIT_IN_SECONDS = 60*60

PREDICTION_MODELS_PROPERTIES = {
                                 'follow_the_leader': {'label': 'Det. Rein. Learning', 'marker': 'd'},
                                 'hedging_method': {'label': 'Ran. Rein. Learning', 'marker': '>'},
                                 }

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
f_location_clusters = analysis_folder%'location_clusters/location_clusters'
f_performance_by_location = analysis_folder%'performance_by_location'                                                                      

us_boundary = [[24.527135,-127.792969], [49.61071,-59.765625]]
south_america_boundary = [[-56.658016,-83.346357], [16.801425,-33.424482]]
eu_boundary = [[34.883261,-16.022138], [71.015901,46.204425]]
sea_boundry = [[-51.291441,92.259112], [28.918782,175.930987]]

def with_gaussian_kde(y_values, x_range = (-1,1,100)):
    density = gaussian_kde(y_values)
    xs = np.linspace(*x_range)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    return xs, density(xs)

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
    def performance_by_location():
        input_file = '%s/prediction_performance_max_time_12/prediction_performance'%dfs_data_folder
        print input_file
        runMRJob(
                 PerformanceByLocation,
                 f_performance_by_location,
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
    def location_clusters():
        runMRJob(
                     LocationClusters,
                     f_location_clusters,
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
#        MRAnalysis.performance_by_location()
#        input_files_start_time, input_files_end_time = \
#                                datetime(2011, 2, 1), datetime(2012, 8, 31)
#        MRAnalysis.hashtags_extractor(input_files_start_time, input_files_end_time)

#        MRAnalysis.propagation_matrix()
#        MRAnalysis.hashtags_with_majority_info()
#        MRAnalysis.hashtags_with_majority_info_at_varying_gaps()
#        MRAnalysis.impact_of_using_locations_to_predict()
#        MRAnalysis.impact_using_mc_simulation()
#        MRAnalysis.gap_occurrence_time_during_hashtag_lifetime()

#        MRAnalysis.location_clusters()

        for i in MRAnalysis.get_input_files(max_time=25):
            print i
        
class PredictHashtagsForLocationsPlots():
    mf_prediction_method_to_properties_dict =\
                                 {
                                   PREDICTION_METHOD_ID_FOLLOW_THE_LEADER : {
                                                                             'label': 'Det. Rein. Learning',
                                                                             'marker': 'o',
                                                                             'color': 'r',
                                                                             },
                                   PREDICTION_METHOD_ID_HEDGING : {
                                                                     'label': 'Ran. Rein. Learning',
                                                                     'marker': '*',
                                                                     'color': 'b',
                                                                 },
                                   PREDICTION_METHOD_ID_LEARNING_TO_RANK : {
                                                                             'label': 'Linear regression',
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
    def perct_of_hashtag_occurrences_vs_hashtag_lifespan():
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.png'
        plt.figure(num=None, figsize=(4.3,3))
        mf_bucket_id_to_perct_occurrences = defaultdict(float)
        for i, data in enumerate(FileIO.iterateJsonFromFile(f_hashtags_extractor, remove_params_dict=True)):
            print i
            mf_bucket_id_to_occurrences_count = defaultdict(float)
            ltuo_occ_time_and_occ_utm_id = data['ltuo_occ_time_and_occ_utm_id']
            occ_times = zip(*ltuo_occ_time_and_occ_utm_id)[0]
            occ_times = map(lambda t: GeneralMethods.approximateEpoch(t, BUCKET_WIDTH), occ_times)
            occ_times = filter_outliers(occ_times)
            occ_times.sort()
            lifespan = occ_times[-1]-occ_times[0]+0.0
            if lifespan>0:
                for occ_time in occ_times: 
                    bucket_id = '%0.02f'%((occ_time-occ_times[0])/lifespan)
                    mf_bucket_id_to_occurrences_count[bucket_id]+=1
                total_occurrences = len(occ_times)+0.0
                for bucket_id, occurrences_count in mf_bucket_id_to_occurrences_count.iteritems():
                    mf_bucket_id_to_perct_occurrences[bucket_id]+=occurrences_count/total_occurrences
        total_perct_value = sum(mf_bucket_id_to_perct_occurrences.values())
        ltuo_bucket_id_and_perct_occurrences = map(
                                                   lambda (b, p): (float(b), p/total_perct_value),
                                                   mf_bucket_id_to_perct_occurrences.iteritems()
                                                )
        ltuo_bucket_id_and_perct_occurrences.sort(key=itemgetter(0))
        bucket_ids, perct_occurrences = zip(*ltuo_bucket_id_and_perct_occurrences)
        perct_occurrences = getCumulativeDistribution(perct_occurrences)
        plt.grid(True)
        ax = plt.subplot(111)
#        ax.set_xscale('log')
        plt.ylim(ymax=1.1, ymin=-0.2)
        plt.xlabel('Hashtag lifespan')
        plt.ylabel('% of hashtag occurrences')
        plt.plot(bucket_ids, perct_occurrences, c='k')
        plt.scatter(bucket_ids, perct_occurrences, c='k')
        savefig(output_file)
    @staticmethod
    def cdf_of_hastag_lifespans():
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.png'
        plt.figure(num=None, figsize=(4.3,3))
        mf_bucket_id_to_count = defaultdict(float)
        for i, data in enumerate(FileIO.iterateJsonFromFile(f_hashtags_extractor, remove_params_dict=True)):
            print i
            ltuo_occ_time_and_occ_utm_id = data['ltuo_occ_time_and_occ_utm_id']
            occ_times = zip(*ltuo_occ_time_and_occ_utm_id)[0]
            occ_times = map(lambda t: GeneralMethods.approximateEpoch(t, BUCKET_WIDTH), occ_times)
            occ_times = filter_outliers(occ_times)
            occ_times = list(set(occ_times))
            occ_times.sort()
            time_diff = occ_times[-1]-occ_times[0]
            mf_bucket_id_to_count[time_diff/(60)]+=1
        total_occurrences = sum(mf_bucket_id_to_count.values())
        ltuo_bucket_id_and_prect_of_hashtags = [(i, j/total_occurrences) for (i,j) in mf_bucket_id_to_count.iteritems()]
        ltuo_bucket_id_and_prect_of_hashtags.sort(key=itemgetter(0))
        bucket_ids, perct_of_hashtags = zip(*ltuo_bucket_id_and_prect_of_hashtags)
        perct_of_hashtags = getCumulativeDistribution(perct_of_hashtags)
        plt.grid(True)
        ax = plt.subplot(111)
        ax.set_xscale('log')
        plt.xlabel('Hashtag lifespans (minutes)')
        plt.ylabel('CDF')
        plt.plot(bucket_ids, perct_of_hashtags, c='k')
        plt.scatter(bucket_ids, perct_of_hashtags, c='k')
        savefig(output_file)
    @staticmethod
    def perct_of_hashtag_occurrences_vs_time_of_propagation():
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.png'
        plt.figure(num=None, figsize=(4.3,3))
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
            Percentage of hashtags >5 locations 0.60316060272
        '''
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.png'
        propagation_distribution = []
        plt.figure(num=None, figsize=(4.3,3))
        for data in FileIO.iterateJsonFromFile(f_hashtags_with_majority_info):
            ltuo_majority_threshold_bucket_time_and_utm_ids = data['ltuo_majority_threshold_bucket_time_and_utm_ids']
            if ltuo_majority_threshold_bucket_time_and_utm_ids:
#                propagation_distribution.append(len(zip(*data['ltuo_majority_threshold_bucket_time_and_utm_ids'])[1]))
                propagation_distribution.append(
                            len(set(list(chain(*zip(*data['ltuo_majority_threshold_bucket_time_and_utm_ids'])[1]))))
                        )
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
        print 'Percentage of hashtags >5 locations', dict(zip(num_of_utms, count_dist))[5]
        plt.plot(num_of_utms, count_dist, c = 'k')
        plt.scatter(num_of_utms, count_dist, c = 'k')
        plt.grid(True)
        plt.xlabel('Number of locations')
        plt.ylabel('CCDF of locations')
        savefig(output_file)
    @staticmethod
    def perct_of_locations_vs_hashtag_lifespan():
        '''
        Percentage of locations propagated in first 6 hours:  0.658433622539
        Percentage of locations between 1 and 6 hours:  0.319961439189
        '''
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.png'
        plt.figure(num=None, figsize=(4.3,3))
        mf_bucket_id_to_perct_of_utm_ids = defaultdict(float)
        for data in FileIO.iterateJsonFromFile(f_hashtags_with_majority_info):
#            print data['hashtag']
            if data['ltuo_majority_threshold_bucket_time_and_utm_ids']:
                ltuo_majority_threshold_bucket_time_and_utm_ids =\
                                                                data['ltuo_majority_threshold_bucket_time_and_utm_ids']
#                majority_threshold_bucket_time = zip(*ltuo_majority_threshold_bucket_time_and_utm_ids)[0]
#                majority_threshold_bucket_time = filter_outliers(majority_threshold_bucket_time)
#                ltuo_majority_threshold_bucket_time_and_utm_ids = filter(
#                                                                 lambda (t, _): t in majority_threshold_bucket_time,
#                                                                 ltuo_majority_threshold_bucket_time_and_utm_ids
#                                                             )
                ltuo_majority_threshold_bucket_time_and_utm_id_counts =\
                                                                    map(
                                                                        lambda (t, utm_ids): (t, len(utm_ids)),
                                                                        ltuo_majority_threshold_bucket_time_and_utm_ids
                                                                        )
                ltuo_majority_threshold_bucket_time_and_utm_id_counts.sort(key=itemgetter(0))
                majority_threshold_bucket_times, utm_id_counts =\
                                                            zip(*ltuo_majority_threshold_bucket_time_and_utm_id_counts)
                total_utm_ids = sum(utm_id_counts)+0.0
                lifespan = majority_threshold_bucket_times[-1] - majority_threshold_bucket_times[0] + 0.0
                if lifespan>0:
                    for majority_threshold_bucket_time, utm_id_count in\
                            ltuo_majority_threshold_bucket_time_and_utm_id_counts:
                        bucket_id =\
                                '%0.02f'%((majority_threshold_bucket_time-majority_threshold_bucket_times[0])/lifespan)
                        mf_bucket_id_to_perct_of_utm_ids[bucket_id]+=utm_id_count/total_utm_ids
                    
        total_perct_value = sum(mf_bucket_id_to_perct_of_utm_ids.values())
        ltuo_bucket_id_and_perct_utm_ids = map(
                                               lambda (b, p): (float(b), p/total_perct_value),
                                               mf_bucket_id_to_perct_of_utm_ids.iteritems()
                                            )
        ltuo_bucket_id_and_perct_utm_ids.sort(key=itemgetter(0))
        bucket_ids, perct_utm_ids = zip(*ltuo_bucket_id_and_perct_utm_ids)
        perct_utm_ids = getCumulativeDistribution(perct_utm_ids)
        plt.grid(True)
        ax = plt.subplot(111)
        plt.ylim(ymax=1.1, ymin=-0.2)
        plt.xlabel('Hashtag lifespan')
        plt.ylabel('CDF of hashtag locations')
        plt.plot(bucket_ids, perct_utm_ids, c='k')
        plt.scatter(bucket_ids, perct_utm_ids, c='k')
        savefig(output_file)
                
    @staticmethod
    def perct_of_locations_vs_hashtag_propaagation_time():
        '''
        Percentage of locations propagated in first 6 hours:  0.658433622539
        Percentage of locations between 1 and 6 hours:  0.319961439189
        '''
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.png'
        mf_majority_threshold_bucket_time_to_num_of_utm_ids = defaultdict(float)
        plt.figure(num=None, figsize=(4.3,3))
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
        '''
        Percentage of locations for which are random 100 is:  0.773913043478
        Percentage of locations for which are random 50 is:  0.827586206897
        Percentage of locations for which are random 25 is:  0.86862745098
        '''
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
                    ltuo_mean_probability_and_perct_of_locations.append([
                                                                         mean_probability,
                                                                         (1.0 - (i+1)/total_locations)
                                                                        ])
                ltuo_mean_probability_and_perct_of_locations.sort(key=itemgetter(0))
                p_mean_probability, p_perct_of_locations = None, None
                for mean_probability, perct_of_locations in ltuo_mean_probability_and_perct_of_locations:
                    if mean_probability>0.05:
                        print 'Percentage of locations for which are random'+\
                                ' %s is: '%min_common_hashtag, p_perct_of_locations
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
        plt.legend(loc=1)
        plt.grid(True)
        plt.ylim(ymax=1.1, ymin=-0.2)
#        plt.ylim(xmax=1.2)
        plt.xlabel('Probability that hashtags propagation between location pairs is random')
        plt.ylabel('CCDF of location pairs')
#        plt.show()
        savefig(output_file)
    @staticmethod
    def perct_of_hashtag_lifespan_vs_perct_of_hashtag_occurrences():
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.png'
        plt.figure(num=None, figsize=(4.3,3))
        ltuo_perct_of_occurrences_and_perct_of_lifespan = None
        for data in FileIO.iterateJsonFromFile(f_gap_occurrence_time_during_hashtag_lifetime, remove_params_dict=True):
            ltuo_perct_of_occurrences_and_perct_of_lifespan = data
        perct_of_occurrences, perct_of_lifespan = zip(*ltuo_perct_of_occurrences_and_perct_of_lifespan)
        perct_of_lifespan = getCumulativeDistribution(perct_of_lifespan)
        plt.plot(perct_of_occurrences, perct_of_lifespan, lw=2, c='k')
        plt.scatter(perct_of_occurrences, perct_of_lifespan, c='k')
        plt.plot([0.15, 0.15], [-0.1, 1.1], '--', c='m', lw=2)
        plt.grid(True)
        plt.ylim(ymax=1.1, ymin=-0.1)
        plt.xlim(xmax=1.1)
        plt.xlabel('% of hashtag occurrences')
        plt.ylabel('% of hashtag lifespan')
        savefig(output_file)
    @staticmethod
    def temp2():
        ltuo_perct_life_time_to_occurrences_distribution = []
        ltuo_perct_life_time_and_num_of_occurrences = []
        ax = plt.subplot(111)
        for data in FileIO.iterateJsonFromFile(f_gap_occurrence_time_during_hashtag_lifetime, remove_params_dict=True):
            if 'perct_life_time_to_occurrences_distribution'==data['key']:
                ltuo_perct_life_time_to_occurrences_distribution = data['value']
            elif 'ltuo_perct_life_time_and_perct_of_occurrences'==data['key']:
                ltuo_perct_life_time_and_num_of_occurrences = data['value']
        print len(ltuo_perct_life_time_to_occurrences_distribution)
        print len(ltuo_perct_life_time_and_num_of_occurrences)
        
        perct_life_time, occurrences_distribution = zip(*ltuo_perct_life_time_to_occurrences_distribution)
        
        perct_life_time = list(perct_life_time)
        for i in range(len(perct_life_time)):
            if i+1<len(perct_life_time):
                perct_life_time[i]=perct_life_time[i+1]
        plt.plot(perct_life_time, occurrences_distribution)
#        ax.set_xscale('log')
        
        perct_life_time, num_of_occurrences = zip(*ltuo_perct_life_time_and_num_of_occurrences)
        total_num_of_occurrences = sum(num_of_occurrences)+0.0
        num_of_occurrences = getCumulativeDistribution(num_of_occurrences)
        perct_of_occurrences = [c/total_num_of_occurrences for c in num_of_occurrences]
        plt.figure()
        ax = plt.subplot(111)
        plt.plot(perct_life_time, perct_of_occurrences)
#        ax.set_xscale('log')
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
    def location_clusters():
        mf_utm_id_to_place_name = dict([
                                        ("-30.0559924674_-51.1556120628_10000", 'Porto Alegre'),
                                        ("-12.8897687915_-38.4930422609_10000", 'Salvador'),
                                        ("-25.531963034_-49.2583718586_10000", 'Curitiba'),
                                        ("-5.83096584727_-35.2125855218_10000", 'Natal'),
                                        ("-8.00174157684_-34.9506560849_10000", 'Recife'),
                                        ("-22.9158446201_-43.4885906927_10000", 'Rio de Janeiro'),
                                        ("-15.7661922583_-47.7531371886_10000", 'Brasilia'),
                                        ("-23.7275399881_-46.6187056015_10000", 'Sao Paulo'),
                                        ("-19.93859689_-43.9966541911_10000", 'Belo Horizonte'),
                                        ("39.4051816666_-0.386807374139_10000", 'Valencia'),
                                        ("41.6675457034_-0.897885246668_10000", 'Zaragoza'),
                                        ("40.5138994175_-3.64922300725_10000", 'Madrid'),
                                        ("36.8081626337_-4.73760992497_10000", 'Malaga'),
                                        ("37.4167203193_-5.88142121144_10000", 'Sevilla'),
                                        ("40.4238144982_-3.64835608078_10000", 'Madrid'),
                                        ('41.4130525526_2.10259967743_10000', 'Barcelona'),
                                        ('41.4139235542_2.22224165917_10000', 'Barcelona'),
                                        ('37.4139133643_-5.99427966323_10000', 'Sevilla'),
                                        ('39.4951942073_-0.383438157077_10000', 'Valencia'),
                                        ])
        graph = nx.Graph()
        for data in FileIO.iterateJsonFromFile(f_location_clusters, remove_params_dict=True):
            u = mf_utm_id_to_place_name.get(data['utm_id'], data['utm_id'])
            v = mf_utm_id_to_place_name.get(data['neighbor_utm_id'], data['neighbor_utm_id'])
            if not graph.has_node(u): 
                graph.add_node(u, {'co-ordinates': UTMConverter.getLatLongUTMIdInLatLongForm(data['utm_id'])})
            if not graph.has_node(v): 
                graph.add_node(v    , {'co-ordinates': UTMConverter.getLatLongUTMIdInLatLongForm(data['neighbor_utm_id'])})
            graph.add_edge(u, v, {'w': data['num_common_hashtags']})
        _, ltuo_node_and_cluster_id = clusterUsingAffinityPropagation(graph)
        ltuo_cluster_id_and_cluster =  [(c_id, zip(*nodes)[0]) 
                        for c_id, nodes in GeneralMethods.group_items_by(ltuo_node_and_cluster_id, key=itemgetter(1))]
#        ltuo_cluster_id_and_cluster = list(enumerate(clusterUsingMCLClustering(graph)))
        ltuo_cluster_id_and_cluster.sort(key=lambda (c, cl): len(cl), reverse=True)
        points, edges, colors, pointLabels = [], [], [], []
        for cluster_id, cluster in ltuo_cluster_id_and_cluster:
            cluster = filter(lambda c: '_' not in c, cluster)
            sub_graph = graph.subgraph(cluster)
            cluster_color = '#FFB5ED'
            labels = sub_graph.nodes()
            if 'Natal' in labels or 'Sevilla' in labels or 'Rio de Janeiro' in labels:
                print 'Ploting: ', labels
                points+=map(itemgetter('co-ordinates'), zip(*sub_graph.nodes_iter(data=True))[1])
                colors+=[cluster_color for i in range(sub_graph.number_of_nodes())]
                pointLabels+=labels
                for u, v in sub_graph.edges_iter(): edges.append((u, v))
        _, m = plotPointsOnWorldMap(
                                        points,
                                        c=colors,
                                        bkcolor = '#ffffff',
                                        resolution = 'h',
                                        s=80,
                                        lw=0,
                                        returnBaseMapObject=True,
                                        pointLabels=pointLabels,
                                        pointLabelColor = 'k',
                                        pointLabelSize=30
                                    )
        for u, v in edges:
            u = graph.node[u]['co-ordinates']
            v = graph.node[v]['co-ordinates']
            m.drawgreatcircle(u[1], u[0], v[1], v[0], color=cluster_color, alpha=1.0, lw=2)
        plt.show()
        
    @staticmethod
    def run():
        PredictHashtagsForLocationsPlots.performance_by_varying_num_of_hashtags()
        PredictHashtagsForLocationsPlots.performance_by_varying_prediction_time_interval()
        PredictHashtagsForLocationsPlots.performance_by_varying_historical_time_interval()
#        PredictHashtagsForLocationsPlots.ccdf_num_of_utmids_where_hashtag_propagates()

#        PredictHashtagsForLocationsPlots.perct_of_hashtag_occurrences_vs_time_of_propagation()
#        PredictHashtagsForLocationsPlots.perct_of_hashtag_occurrences_vs_hashtag_lifespan()

#        PredictHashtagsForLocationsPlots.perct_of_locations_vs_hashtag_propaagation_time()
#        PredictHashtagsForLocationsPlots.perct_of_locations_vs_hashtag_lifespan()

#        PredictHashtagsForLocationsPlots.perct_of_locations_vs_hashtag_propaagation_time_at_varying_gaps()
#        PredictHashtagsForLocationsPlots.impact_of_using_location_to_predict_hashtag()

#        PredictHashtagsForLocationsPlots.impact_of_using_location_to_predict_hashtag_with_mc_simulation_gaussian_kde()
#        PredictHashtagsForLocationsPlots.impact_of_using_location_to_predict_hashtag_with_mc_simulation_cdf()
#        PredictHashtagsForLocationsPlots.impact_of_using_location_to_predict_hashtag_with_mc_simulation_cdf_multi()
#        PredictHashtagsForLocationsPlots.impact_of_using_location_to_predict_hashtag_with_mc_simulation_examples()

#        PredictHashtagsForLocationsPlots.example_of_hashtag_propagation_patterns()
            
#        PredictHashtagsForLocationsPlots.perct_of_hashtag_lifespan_vs_perct_of_hashtag_occurrences()
#        PredictHashtagsForLocationsPlots.cdf_of_hastag_lifespans()

#        PredictHashtagsForLocationsPlots.temp1()

#        PredictHashtagsForLocationsPlots.location_clusters()

class PerformanceByLocationAnalysis(object):
    @staticmethod
    def location_distribution():
        def get_boundary(location):
            for id, boundary in zip(
                                            ['us', 'sa', 'eu', 'sea'],
                                            [us_boundary, south_america_boundary, eu_boundary, sea_boundry]
                                        ):
                        if isWithinBoundingBox(location, boundary): return id
                        
        raw_data = list(FileIO.iterateJsonFromFile(f_performance_by_location, True))
        getLocation = lambda lid: getLocationFromLid(lid.replace('_', ' '))
        locations = map(getLocation, map(itemgetter('location'), raw_data))
        boundaries = filter(lambda b: b, map(get_boundary, locations))
        mf_boundary_to_count = defaultdict(float)
        for boundary in boundaries: mf_boundary_to_count[boundary]+=1
        total = sum(mf_boundary_to_count.values())
        ltuo_boundary_and_count = sorted(mf_boundary_to_count.items(), key=itemgetter(0))
        print '\t'.join(list(zip(*ltuo_boundary_and_count)[0]))
        print '\t'.join(map(str, map(lambda c: c/total, zip(*ltuo_boundary_and_count)[1])))
    @staticmethod
    def model_distribution():
        def plot_distribution(key):
            ltuo_model_and_score = map(itemgetter(key), performances)
            models = [sorted(lt_m_and_s, key=itemgetter(1))[-1][0] for lt_m_and_s in ltuo_model_and_score]
            total = len(models) + 0.0
            ltuo_model_and_value = [(m, len(list(l_items))/total) 
                                                for m, l_items in GeneralMethods.group_items_by(models, key=lambda i:i)]
            print key
            values = zip(*ltuo_model_and_value)[1]
            print '\t'.join(['overall'] + map(str, list(values)))
            ltuo_location_and_model = zip(locations, models)
            mf_boundary_to_models= defaultdict(dict)
            for location, model in ltuo_location_and_model:
                for id, boundary in zip(
                                        ['us', 'sa', 'eu', 'sea'],
                                        [us_boundary, south_america_boundary, eu_boundary, sea_boundry]
                                    ):
                    if isWithinBoundingBox(location, boundary):
                        if model not in mf_boundary_to_models[id]: mf_boundary_to_models[id][model]=0.0
                        mf_boundary_to_models[id][model]+=1
                        break
            
            for boundary, mf_model_to_count in mf_boundary_to_models.iteritems():
                total = sum(mf_model_to_count.values()) + 0.0
                ltuo_model_and_value = sorted([(m, c/total)for m, c in mf_model_to_count.items()], key=itemgetter(0))
                values = zip(*ltuo_model_and_value)[1]
                print '\t'.join([boundary] + map(str, list(values)))
            print '\t'.join(list(zip(*ltuo_model_and_value)[0]))
        raw_data = list(FileIO.iterateJsonFromFile(f_performance_by_location, True))
        getLocation = lambda lid: getLocationFromLid(lid.replace('_', ' '))
        locations = map(getLocation, map(itemgetter('location'), raw_data))
        performances = map(itemgetter('performance_summary'), raw_data)
        plot_distribution('impact')
        plot_distribution('accuracy')
    @staticmethod
    def metric_distribution():
        output_file_format = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'/%s.png'
        def plot_distribution(key):
            plt.figure(num=None, figsize=(4.3,3))
            ltuo_model_and_score = map(itemgetter(key.lower()), performances)
            scores = [sorted(lt_m_and_s, key=itemgetter(1))[-1][1] for lt_m_and_s in ltuo_model_and_score]
            values, bins = np.histogram(scores, bins=20)
            values, bins = list(values), list(bins[:-1])
            total = sum(values) + 0.0
            bins = map(lambda v: v+0.025, bins)
            values = map(lambda v: v/total, values)
            plt.plot(bins, values, c='k', lw=2)
            plt.scatter(bins, values, c='k')
            plt.grid(True)
            plt.xlabel(key)
            plt.ylabel('% of locations')
            savefig(output_file_format%(GeneralMethods.get_method_id()+ '_' +key))
        def plot_cdf(key):
            plt.figure(num=None, figsize=(4.3,3))
            ltuo_model_and_score = map(itemgetter(key.lower()), performances)
            scores = [sorted(lt_m_and_s, key=itemgetter(1))[-1][1] for lt_m_and_s in ltuo_model_and_score]
            values, bins = np.histogram(scores, bins=20)
            values, bins = list(values), list(bins[:-1])
            total = sum(values) + 0.0
            bins = map(lambda v: v+0.025, bins)
            values = map(lambda v: v/total, values)
            values = getInverseCumulativeDistribution(values)
            plt.plot(bins, values, c='k', lw=2)
            plt.scatter(bins, values, c='k')
            plt.grid(True)
            plt.xlabel(key)
            plt.ylabel('CCDF of locations')
            savefig(output_file_format%(GeneralMethods.get_method_id()+ '_' +key))
        performances = map(
                           itemgetter('performance_summary'),
                           FileIO.iterateJsonFromFile(f_performance_by_location, True)
                           )
        plot_distribution('Impact')
        plot_distribution('Accuracy')
        plot_cdf('Impact')
        plot_cdf('Accuracy')
    @staticmethod
    def geo_area_specific_distribution():
        output_file_format = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'/%s.png'
        def plot_distribution(key, locations):
            plt.figure(num=None, figsize=(6,3))
            plt.subplots_adjust(bottom=0.2, top=0.9)
            plt.subplot(111)
            ltuo_model_and_score = map(itemgetter(key), performances)
            scores = [sorted(lt_m_and_s, key=itemgetter(1))[-1][1] for lt_m_and_s in ltuo_model_and_score]
            ltuo_location_and_score = zip(locations, scores)
            mf_us_boundary_to_scores = defaultdict(list)
            for location, score in ltuo_location_and_score:
                for id, boundary in zip(
                                        ['us', 'sa', 'eu', 'sea'],
                                        [us_boundary, south_america_boundary, eu_boundary, sea_boundry]
                                    ):
                    if isWithinBoundingBox(location, boundary):
                        mf_us_boundary_to_scores[id].append(score)
                        break
            values, bins = np.histogram(scores, bins=20)
            values, bins = list(values), list(bins[:-1])
            total = sum(values)+0.0
            values = map(lambda v: v/total, values)
            values = getInverseCumulativeDistribution(values)
            plt.plot(bins, values, label='WW')
#            PerformanceByLocationAnalysis.plot_distribution(bins, values, output_file_format%(key+'_global'))
            for boundary, scores_b in mf_us_boundary_to_scores.iteritems():
                values, bins = np.histogram(scores_b, bins=20)
                values, bins = list(values), list(bins[:-1])
                total = sum(values) + 0.0
                values = map(lambda v: v / total, values)
                values = getInverseCumulativeDistribution(values)
#                PerformanceByLocationAnalysis.plot_distribution(
#                                                                bins, 
#                                                                values, 
#                                                                output_file_format % (key + '_' + boundary)
#                                                                )
                plt.plot(bins, values, label=boundary)
            plt.legend(loc=3)
            plt.xlim(xmin=-0.1, xmax=1.2)
            plt.ylim(ymin=-0.1, ymax=1.2)
            plt.ylabel('Distribution')
            plt.xlabel('Score')
            plt.grid(True)
            savefig(output_file_format%key)
        raw_data = list(FileIO.iterateJsonFromFile(f_performance_by_location, True))
        getLocation = lambda lid: getLocationFromLid(lid.replace('_', ' '))
        locations = map(getLocation, map(itemgetter('location'), raw_data))
        performances = map(itemgetter('performance_summary'), raw_data)
        plot_distribution('impact', locations)
        plot_distribution('accuracy', locations)
    @staticmethod
    def plot_distribution(bins, values, output_file):
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9)
        plt.subplot(111)
        plt.plot(bins, values, lw=2, c='k')
        plt.scatter(bins, values)
        plt.xlim(xmin=-0.05, xmax=1.0)
        plt.ylim(ymin=-0.05, ymax=0.30)
        plt.ylabel('Distribution')
        plt.xlabel('Score')
        plt.grid(True)
        savefig(output_file)
    @staticmethod
    def top_and_bottom_locations():
        def get_top_and_bottom_locations(key):
            ltuo_model_and_score = map(itemgetter(key), performances)
            scores = [sorted(lt_m_and_s, key=itemgetter(1))[-1][1] for lt_m_and_s in ltuo_model_and_score]
            ltuo_location_and_score = zip(locations, scores)
            mf_boundary_to_ltuo_location_and_score = defaultdict(list)
            for location, score in ltuo_location_and_score:
                for id, boundary in zip(
                                        ['us', 'sa', 'eu', 'sea'],
                                        [us_boundary, south_america_boundary, eu_boundary, sea_boundry]
                                    ):
                    if isWithinBoundingBox(location, boundary):
                        mf_boundary_to_ltuo_location_and_score[id].append((location, score))
                        break
            for boundary, ltuo_location_and_score in mf_boundary_to_ltuo_location_and_score.iteritems():
                ltuo_location_and_score.sort(key=itemgetter(1))
                print boundary
                print ltuo_location_and_score[-5:]
                print ltuo_location_and_score[:5]
                exit()
        raw_data = list(FileIO.iterateJsonFromFile(f_performance_by_location, True))
        getLocation = lambda lid: getLocationFromLid(lid.replace('_', ' '))
        locations = map(getLocation, map(itemgetter('location'), raw_data))
        performances = map(itemgetter('performance_summary'), raw_data)
        get_top_and_bottom_locations('impact')
#        get_top_and_bottom_locations('accuracy', locations)
    @staticmethod
    def learner_flipping_time_series(learning_types, no_of_hashtags):
        plt.figure(num=None, figsize=(6.,3))
        plt.subplots_adjust(bottom=0.2, top=0.9)
        plt.subplot(111)
        for learning_type in learning_types:
            input_weight_file = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/'+\
                            'testing/models/2011-09-01_2011-11-01/30_60/%s/%s_weights'%(no_of_hashtags, learning_type)
            map_from_ep_time_unit_to_no_of_locations_that_didnt_flip = {}
            map_from_location_to_previously_selected_model = {}
            for data in FileIO.iterateJsonFromFile(input_weight_file, True):
                if data['metricId']=='accuracy':
                    map_from_location_to_map_from_model_to_weight = data['location_weights']
                    ep_time_unit = data['tu']
                    no_of_locations_that_didnt_flip = 0.0 
                    for location, map_from_model_to_weight in map_from_location_to_map_from_model_to_weight.iteritems():
                        model_selected = MAP_FROM_MODEL_TO_MODEL_TYPE\
                            [LearningAnalysis.MAP_FROM_LEARNING_TYPE_TO_MODEL_SELECION_METHOD[learning_type]\
                                                                                            (map_from_model_to_weight)]
                        if location in map_from_location_to_previously_selected_model and \
                                            map_from_location_to_previously_selected_model[location]==model_selected: 
                            no_of_locations_that_didnt_flip+=1
                        map_from_location_to_previously_selected_model[location] = model_selected
                    map_from_ep_time_unit_to_no_of_locations_that_didnt_flip[ep_time_unit] = \
                                                                                        no_of_locations_that_didnt_flip
            total_no_of_locations = len(map_from_location_to_previously_selected_model)
            tuples_of_ep_time_unit_and_percentage_of_locations_that_flipped = [
                (ep_time_unit, 
                 1.0 - (
                        map_from_ep_time_unit_to_no_of_locations_that_didnt_flip[ep_time_unit]/total_no_of_locations)) 
                                   for ep_time_unit in sorted(map_from_ep_time_unit_to_no_of_locations_that_didnt_flip)
                                                                            ]
            ep_first_time_unit = tuples_of_ep_time_unit_and_percentage_of_locations_that_flipped[0][0]
            x_data, y_data = zip(*tuples_of_ep_time_unit_and_percentage_of_locations_that_flipped)
            x_data, y_data = splineSmooth(x_data, y_data)
            plt.plot(
                        [(x-ep_first_time_unit)/(60*60) for x in x_data], 
                        y_data,
                        c=MAP_FROM_MODEL_TO_COLOR[learning_type],
                        lw=2,
                        label=PREDICTION_MODELS_PROPERTIES[learning_type]['label'],
#                        marker=MAP_FROM_MODEL_TO_MARKER[learning_type]
                     )
        plt.grid(True)
        plt.legend()
#        leg = plt.legend(loc=1, ncol=1, fancybox=True)
#        leg.get_frame().set_alpha(0.5)
        plt.xlabel('Time (hours)'), plt.ylabel('% of locations that flipped')
        savefig(fld_google_drive_data_analysis%'learner_flipping_time_series.png')    
    @staticmethod
    def flipping_ratio_correlation_with_no_of_occurrences_at_location(learning_types, no_of_hashtags):
        plt.figure(num=None, figsize=(6,3))
        plt.subplots_adjust(bottom=0.2, top=0.9)
        plt.subplot(111)
        for learning_type in learning_types:
            NO_OF_OCCURRENCES_BIN_SIZE= 200
            # Load flipping ratio data.
            map_from_location_to_flipping_ratio = dict(LearningAnalysis._get_flipping_ratio_for_all_locations(learning_type, no_of_hashtags))
            # Load no. of occurrences data
            #        startTime, endTime, outputFolder = datetime(2011, 9, 1), datetime(2011, 11, 1), 'testing'
            startTime, endTime, outputFolder = datetime(2011, 4, 1), datetime(2012, 1, 31), 'complete' # Complete duration
            input_file = timeUnitWithOccurrencesFile%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
            map_from_location_to_no_of_occurrences_at_location = defaultdict(float)
            for time_unit_object in FileIO.iterateJsonFromFile(input_file, True):
                for (_, location, _) in time_unit_object['oc']: 
                    if location in map_from_location_to_flipping_ratio:
                        map_from_location_to_no_of_occurrences_at_location[location]+=1
            tuples_of_location_and_flipping_ratio_and_no_of_occurrences_at_location = [(
                                                                                        location, 
                                                                                        map_from_location_to_flipping_ratio[location],
                                                                                        map_from_location_to_no_of_occurrences_at_location[location],
                                                                                        )
                                                                                       for location in map_from_location_to_no_of_occurrences_at_location]
            # Filter locations for no. of occurrences.
            no_of_occurrences_at_location = zip(*tuples_of_location_and_flipping_ratio_and_no_of_occurrences_at_location)[2]
            print len(tuples_of_location_and_flipping_ratio_and_no_of_occurrences_at_location)
            _, upper_range_no_of_occurrences_at_location = getOutliersRangeUsingIRQ(no_of_occurrences_at_location)
            tuples_of_location_and_flipping_ratio_and_no_of_occurrences_at_location = filter(lambda (_,__,no_of_occurrences_at_location): 
                                                                                                no_of_occurrences_at_location<=upper_range_no_of_occurrences_at_location,
                                                                                             tuples_of_location_and_flipping_ratio_and_no_of_occurrences_at_location)
            print len(tuples_of_location_and_flipping_ratio_and_no_of_occurrences_at_location)
            # Bin no. of occurrences.
            map_from_no_of_occurrences_at_location_bin_to_flipping_ratios = defaultdict(list)
            for _, flipping_ratio, no_of_occurrences_at_location in tuples_of_location_and_flipping_ratio_and_no_of_occurrences_at_location:
                no_of_occurrences_at_location_bin = int(no_of_occurrences_at_location/NO_OF_OCCURRENCES_BIN_SIZE)*NO_OF_OCCURRENCES_BIN_SIZE + NO_OF_OCCURRENCES_BIN_SIZE + 0.0
                map_from_no_of_occurrences_at_location_bin_to_flipping_ratios[no_of_occurrences_at_location_bin].append(flipping_ratio)
            for no_of_occurrences_at_location_bin in sorted(map_from_no_of_occurrences_at_location_bin_to_flipping_ratios):
                flipping_ratios = map_from_no_of_occurrences_at_location_bin_to_flipping_ratios[no_of_occurrences_at_location_bin]
                flipping_ratios = filter_outliers(flipping_ratios)
                if len(flipping_ratios) >= 5: map_from_no_of_occurrences_at_location_bin_to_flipping_ratios[no_of_occurrences_at_location_bin] = flipping_ratios
                else: del map_from_no_of_occurrences_at_location_bin_to_flipping_ratios[no_of_occurrences_at_location_bin]
            # Plot data.
#            for no_of_occurrences_at_location_bin, flipping_ratios in \
#                  sorted(map_from_no_of_occurrences_at_location_bin_to_flipping_ratios.iteritems(), key=itemgetter(0)):
#                print no_of_occurrences_at_location_bin, len(flipping_ratios)
            x_no_of_occurrences_at_location_bins, y_mean_flipping_ratios = zip(*[ (no_of_occurrences_at_location_bin, np.mean(flipping_ratios)) 
                  for no_of_occurrences_at_location_bin, flipping_ratios in 
                  sorted(map_from_no_of_occurrences_at_location_bin_to_flipping_ratios.iteritems(), key=itemgetter(0))
#                  if len(flipping_ratios) > 10
                  ])
            pearsonCoeff, p_value = scipy.stats.pearsonr(x_no_of_occurrences_at_location_bins, y_mean_flipping_ratios)
            print round(pearsonCoeff,2), round(p_value, 2)
            x_no_of_occurrences_at_location_bins, y_mean_flipping_ratios = np.array(list(x_no_of_occurrences_at_location_bins)), np.array(list(y_mean_flipping_ratios))
            parameters_after_fitting = CurveFit.getParamsAfterFittingData(x_no_of_occurrences_at_location_bins, y_mean_flipping_ratios, CurveFit.lineFunction, [0., 0.])
            y_fitted_mean_flipping_ratios = CurveFit.getYValues(CurveFit.lineFunction, parameters_after_fitting, x_no_of_occurrences_at_location_bins)
            plt.scatter(x_no_of_occurrences_at_location_bins, y_mean_flipping_ratios, lw=0, c='k', marker=MAP_FROM_MODEL_TO_MARKER[learning_type])
            plt.plot(x_no_of_occurrences_at_location_bins, y_fitted_mean_flipping_ratios, lw=1, c='k')
        plt.grid(True)
#        plt.legend()
        plt.xlim(xmin=0.0)
        plt.xlabel('Hastag density'), plt.ylabel('Flipping ratio')
        plt.legend()
        savefig(fld_google_drive_data_analysis%'flipping_ratio_correlation_with_no_of_occurrences_at_location.png')    
#        file_learning_analysis = './images/%s.png'%GeneralMethods.get_method_id()
#        FileIO.createDirectoryForFile(file_learning_analysis)
#        plt.show()
#        plt.savefig(file_learning_analysis)
#        plt.clf()
    @staticmethod
    def run():
#        PerformanceByLocationAnalysis.location_distribution()
#        PerformanceByLocationAnalysis.model_distribution()
#        PerformanceByLocationAnalysis.metric_distribution()
#        PerformanceByLocationAnalysis.geo_area_specific_distribution()
#        PerformanceByLocationAnalysis.top_and_bottom_locations()
        PerformanceByLocationAnalysis.learner_flipping_time_series(
                                                                   [
                                                                    ModelSelectionHistory.FOLLOW_THE_LEADER,
                                                                    ModelSelectionHistory.HEDGING_METHOD
                                                                    ],
                                                                   4)
        PerformanceByLocationAnalysis.flipping_ratio_correlation_with_no_of_occurrences_at_location(
                                                                   [
                                                                    ModelSelectionHistory.FOLLOW_THE_LEADER,
#                                                                    ModelSelectionHistory.HEDGING_METHOD
                                                                    ],
                                                                   4)
        
if __name__ == '__main__':
#    MRAnalysis.run()
    PredictHashtagsForLocationsPlots.run()
#    PerformanceByLocationAnalysis.run()
