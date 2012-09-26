'''
Created on Sep 26, 2012

@author: krishnakamath
'''
from mr_predict_hashtags_for_locations import PredictingHastagsForLocations
from library.mrjobwrapper import runMRJob
from library.file_io import FileIO

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
        for data in FileIO.iterateJsonFromFile(f_prediction_performance):
            print data.keys()
    @staticmethod
    def run():
        Plots.temp()
        
if __name__ == '__main__':
#    MRAnalysis.run()
    Plots.run()
