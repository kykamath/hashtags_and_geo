'''
Created on Sep 26, 2012

@author: krishnakamath
'''
from datetime import timedelta
from itertools import groupby
from library.mrjobwrapper import runMRJob
from library.file_io import FileIO
#from mr_predict_hashtags_for_locations import PredictingHastagsForLocations
from operator import itemgetter

TIME_UNIT_IN_SECONDS = 60*60

dfs_data_folder = 'hdfs:///user/kykamath/geo/hashtags/'
analysis_folder = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/predict_hashtags_for_locations/%s'

f_prediction_performance = analysis_folder%'prediction_performance'

class MRAnalysis():
    @staticmethod
    def get_input_files(min_time = 1, max_time=25):
        range_1 = [(i,1)for i in range(min_time, max_time-1)]
        range_2 = [(1,i)for i in range(min_time, max_time-1)]
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
                 ['hdfs:///user/kykamath/geo/hashtags/linear_regression'],
                 jobconf={'mapred.reduce.tasks':10}
                 )
    @staticmethod
    def run():
#        MRAnalysis.experiments()
        MRAnalysis.get_input_files()
        
class Plots():
    @staticmethod
    def temp():
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
            ltuo_prediction_method_and_num_of_hashtags_and_metric_value.sort(key=itemgetter(0))
            
    @staticmethod
    def run():
        Plots.temp()
        
if __name__ == '__main__':
    MRAnalysis.run()
#    Plots.run()
