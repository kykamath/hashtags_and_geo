'''
Created on Jan 22, 2013

@author: krishnakamath
'''
import sys
sys.path.append('../')

from library.mrjobwrapper import runMRJob
from mr_predict_hashtags_for_locations_jan_13 import HashtagsByModelsByLocations


dfs_input = 'hdfs:///user/kykamath/geo/hashtags/2011-09-01_2011-11-01/360_120/100/linear_regression'

analysis_folder = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/predict_hashtags_for_locations/%s'
f_hashtags_by_models_by_locations = analysis_folder%'hashtags_by_models_by_locations'

class MRAnalysis(object):
    @staticmethod
    def hashtags_by_models_by_locations():
        runMRJob(
                     HashtagsByModelsByLocations,
                     f_hashtags_by_models_by_locations,
                     [dfs_input],
                     jobconf={'mapred.reduce.tasks':500, 'mapred.task.timeout': 86400000}
                 )
    @staticmethod
    def run():
        MRAnalysis.hashtags_by_models_by_locations()

if __name__ == '__main__':
    MRAnalysis.run()