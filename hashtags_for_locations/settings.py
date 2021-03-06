'''
Created on Feb 12, 2012

@author: kykamath
'''
US_BOUNDARY = [[24.527135,-127.792969], [49.61071,-59.765625]]
PARTIAL_WORLD_BOUNDARY = [[-58.447733,-153.457031], [72.127936,75.410156]]

hdfsInputFolder = 'hdfs:///user/kykamath/geo/hashtags/%s/'

analysisFolder = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/%s/'
#analysisFolder = '/data/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/%s/'
hashtagsWithoutEndingWindowFile = analysisFolder+'%s_%s/hashtagsWithoutEndingWindow'
hashtagsWithoutEndingWindowWithoutLatticeApproximationFile = analysisFolder+'%s_%s/hashtagsWithoutEndingWindowWithoutLatticeApproximation'
hashtagsWithEndingWindowFile = analysisFolder+'%s_%s/hashtagsWithEndingWindow'
hashtagsAllOccurrencesWithinWindowFile = analysisFolder+'%s_%s/hashtagsAllOccurrencesWithinWindow'
locationsWithOccurrencesFile = analysisFolder+'%s_%s/locationsWithOccurrences'
timeUnitWithOccurrencesFile = analysisFolder+'%s_%s/timeUnitWithOccurrences'
latticeGraphFile = analysisFolder+'%s_%s/latticeGraph'
#modelsFolder = analysisFolder+'models/'
modelsFolder = analysisFolder+'models_1/'
locationsGraphFile = '/mnt/chevron/kykamath/data/geo/hashtags/analysis/training_world/2_11/latticeGraph'

data_analysis_folder = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/data_analysis/%s/'


