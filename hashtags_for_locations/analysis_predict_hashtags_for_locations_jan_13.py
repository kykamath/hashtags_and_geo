'''
Created on Jan 22, 2013

@author: krishnakamath
'''
#import sys
#sys.path.append('../')

from datetime import datetime
from datetime import timedelta
from library.mrjobwrapper import runMRJob
from mr_predict_hashtags_for_locations_jan_13 import HashtagsByModelsByLocations

TIME_UNIT_IN_SECONDS = 60*60

dfs_data_folder = 'hdfs:///user/kykamath/geo/hashtags/2011-09-01_2011-11-01/'
dfs_input = dfs_data_folder + '360_120/100/linear_regression'

analysis_folder = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/predict_hashtags_for_locations/%s'
f_hashtags_by_models_by_locations = analysis_folder%'hashtags_by_models_by_locations'

class MRAnalysis(object):
    @staticmethod
    def get_input_files():
        for j in range(1, 7):
            predictionTimeInterval = timedelta(seconds=j*TIME_UNIT_IN_SECONDS)
            yield '%s2011-09-01_2011-11-01/%s_%s/100/linear_regression'%(
                                                                          dfs_data_folder,
                                                                          360,
                                                                          predictionTimeInterval.seconds/60
                                                                        )
    @staticmethod
    def hashtags_by_models_by_locations():
        runMRJob(
                    HashtagsByModelsByLocations,
                    f_hashtags_by_models_by_locations,
#                     [dfs_input],
                    MRAnalysis.get_input_files(),
                    jobconf={'mapred.reduce.tasks':500, 'mapred.task.timeout': 86400000}
                 )
    @staticmethod
    def run():
        MRAnalysis.hashtags_by_models_by_locations()

if __name__ == '__main__':
    MRAnalysis.run()
    
