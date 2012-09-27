'''
Created on Sep 26, 2012

@author: krishnakamath
'''
from datetime import timedelta
from itertools import groupby
from library.classes import GeneralMethods
from library.file_io import FileIO
from library.mrjobwrapper import runMRJob
from library.plotting import savefig
from mr_predict_hashtags_for_locations import EvaluationMetric
from mr_predict_hashtags_for_locations import PredictingHastagsForLocations
from mr_predict_hashtags_for_locations import PREDICTION_METHOD_ID_FOLLOW_THE_LEADER
from mr_predict_hashtags_for_locations import PREDICTION_METHOD_ID_HEDGING
from mr_predict_hashtags_for_locations import PREDICTION_METHOD_ID_LEARNING_TO_RANK
from operator import itemgetter
import matplotlib.pyplot as plt
import os

TIME_UNIT_IN_SECONDS = 60*60

dfs_data_folder = 'hdfs:///user/kykamath/geo/hashtags/'
analysis_folder = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/predict_hashtags_for_locations/%s'
fld_google_drive_data_analysis = os.path.expanduser('~/Google Drive/Desktop/'\
                                                    'hashtags_and_geo/hashtags_for_locations/%s') 

f_prediction_performance = analysis_folder%'prediction_performance'

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
    def experiments():
        runMRJob(
                 PredictingHastagsForLocations,
                 f_prediction_performance,
                 MRAnalysis.get_input_files(max_time=11),
                 jobconf={'mapred.reduce.tasks':500}
                 )
    @staticmethod
    def run():
        MRAnalysis.experiments()
#        print list(MRAnalysis.get_input_files())
        
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
    def performance_by_varying_number_of_hashtags():
        output_file_format = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'_%s.png'
        performance_data = list(FileIO.iterateJsonFromFile('temp_f'))
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
            plt.xlabel('Number of hashtags (k)')
            plt.legend(loc=4)
            plt.grid(True)
            savefig(output_file_format%metric)
            
    @staticmethod
    def run():
        PredictHashtagsForLocationsPlots.performance_by_varying_number_of_hashtags()
        
if __name__ == '__main__':
    MRAnalysis.run()
#    PredictHashtagsForLocationsPlots.run()
