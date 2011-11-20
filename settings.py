'''
Created on Nov 19, 2011

@author: kykamath
'''

# Boundaries
us_boundary = [[24.527135,-127.792969], [49.61071,-59.765625]]

#Input files
tempInputFile = 'hdfs:///user/kykamath/geo/twitter/2_11'
inputFolder = 'hdfs:///user/kykamath/geo/hashtags/'

# Analysis
hashtagsAnalysisFolder = '/mnt/chevron/kykamath/data/geo/hashtags/analysis/'
hashtagsFile = hashtagsAnalysisFolder+'hashtags'
hashtagsWithoutEndingWindowFile = hashtagsAnalysisFolder+'hashtagsWithoutEndingWindow'
hashtagsAverageHaversineDistanceFile = hashtagsAnalysisFolder+'hashtagsAverageHaversineDistance'
hashtagsDistributionInTimeFile = hashtagsAnalysisFolder+'hashtagsDistributionInTime'
hashtagsDistributionInLatticeFile = hashtagsAnalysisFolder+'hashtagsDistributionInLattice'

# Images
hashtagsImagesFolder = '/data/geo/hashtags/images/'
hashtagsImagesTimeVsDistanceFolder = hashtagsImagesFolder+'time_vs_distance/'