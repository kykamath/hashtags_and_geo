'''
Created on Sep 26, 2012

@author: krishnakamath
'''
from mr_predict_hashtags_for_locations import PredictingHastagsForLocations
from library.mrjobwrapper import runMRJob

dfs_data_folder = 'hdfs:///user/kykamath/geo/hashtags/'
analysis_folder = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/predict_hashtags_for_locations/%s'

f_prediction_performance = analysis_folder%'prediction_performance'

mr_class = PredictingHastagsForLocations
output_file = 'output' 
runMRJob(
         PredictingHastagsForLocations,
         f_prediction_performance,
         ['hdfs:///user/kykamath/geo/hashtags/linear_regression'],
         jobconf={'mapred.reduce.tasks':10}
         )
