'''
Created on Feb 12, 2012

@author: kykamath
'''
hdfsInputFolder = 'hdfs:///user/kykamath/geo/hashtags/%s/'

analysisFolder = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/%s/'
#analysisFolder = '/data/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/%s/'
hashtagsWithoutEndingWindowFile = analysisFolder+'%s_%s/hashtagsWithoutEndingWindow'
hashtagsWithEndingWindowFile = analysisFolder+'%s_%s/hashtagsWithEndingWindow'
hashtagsAllOccurrencesWithinWindowFile = analysisFolder+'%s_%s/hashtagsAllOccurrencesWithinWindow'
locationsWithOccurrencesFile = analysisFolder+'%s_%s/locationsWithOccurrences'
timeUnitWithOccurrencesFile = analysisFolder+'%s_%s/timeUnitWithOccurrences'
modelsFolder = analysisFolder+'models/'

locationsGraphFile = '/mnt/chevron/kykamath/data/geo/hashtags/analysis/training_world/2_11/latticeGraph'

