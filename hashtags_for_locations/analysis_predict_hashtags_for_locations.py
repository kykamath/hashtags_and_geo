'''
Created on Sep 26, 2012

@author: krishnakamath
'''
from itertools import groupby
from library.mrjobwrapper import runMRJob
from library.file_io import FileIO
#from mr_predict_hashtags_for_locations import PredictingHastagsForLocations
from operator import itemgetter

dfs_data_folder = 'hdfs:///user/kykamath/geo/hashtags/'
analysis_folder = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/predict_hashtags_for_locations/%s'

f_prediction_performance = analysis_folder%'prediction_performance'

class MRAnalysis():
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
        MRAnalysis.experiments()
        
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
#    MRAnalysis.run()
    Plots.run()
