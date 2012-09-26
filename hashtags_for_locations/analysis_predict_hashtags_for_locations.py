'''
Created on Sep 26, 2012

@author: krishnakamath
'''
from mr_predict_hashtags_for_locations import PredictingHastagsForLocations
from library.mrjobwrapper import runMRJob

mr_class = PredictingHastagsForLocations
output_file = 'output' 
runMRJob(
         PredictingHastagsForLocations,
         output_file,
         ['hdfs:///user/kykamath/geo/hashtags/linear_regression'],
         jobconf={'mapred.reduce.tasks':10}
         )
