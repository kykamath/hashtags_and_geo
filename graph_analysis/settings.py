'''
Created on Dec 15, 2011

@author: kykamath
'''

# Boundaries
us_boundary = [[24.527135,-127.792969], [49.61071,-59.765625]]

# HDFS
hdfsInputFolder = 'hdfs:///user/kykamath/geo/hashtags/'

# Analysis
hashtagsAnalysisFolder = '/mnt/chevron/kykamath/data/geo/hashtags/temporal_graphs/analysis/'
hashtagsFile = hashtagsAnalysisFolder+'%s/%s/hashtags'
epochGraphsFile = hashtagsAnalysisFolder+'%s/%s/epochGraphs'
tempEpochGraphsFile = hashtagsAnalysisFolder+'%s/%s/tempEpochGraphs'
runningTimesFolder = hashtagsAnalysisFolder+'runningTimes/%s'
qualityMetricsFolder = hashtagsAnalysisFolder+'qualityMetrics/%s'
randomGraphsFolder = hashtagsAnalysisFolder+'random_graphs/%s'
