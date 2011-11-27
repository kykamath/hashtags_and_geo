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
hashtagsWithoutEndingWindowFile = hashtagsAnalysisFolder+'%s/hashtagsWithoutEndingWindow'
hashtagsWithoutEndingWindowUsingBoundaryFile = hashtagsAnalysisFolder+'%s/%s/%s_hashtagsWithoutEndingWindow'
#hashtagsCenterOfMassAnalysisWithoutEndingWindowFile = hashtagsAnalysisFolder+'%s/hashtagsCenterOfMassAnalysisWithoutEndingWindow'
#hashtagsSpreadInTimeFile = hashtagsAnalysisFolder+'%s/hashtagsSpreadInTime'
#hashtagsMeanDistanceInTimeFile = hashtagsAnalysisFolder + '%s/hashtagsMeanDistanceInTime'
hashtagsDisplacementStatsFile = hashtagsAnalysisFolder+'%s/%s/hashtagsDisplacementStats'
hashtagsDistributionInTimeFile = hashtagsAnalysisFolder+'hashtagsDistributionInTime'
hashtagsDistributionInLatticeFile = hashtagsAnalysisFolder+'hashtagsDistributionInLattice'
hashtagsAnalayzeLocalityIndexAtKFile = hashtagsAnalysisFolder+'%s/%s/hashtagsAnalayzeLocalityIndexAtK'
hashtagWithGuranteedSourceFile = hashtagsAnalysisFolder+'%s/%s/hashtagWithGuranteedSource'

# Images
hashtagsImagesFolder = '/data/geo/hashtags/images/'
hashtagsImagesTimeVsDistanceFolder = hashtagsImagesFolder+'time_vs_distance/'
hashtagsImagesCenterOfMassFolder = hashtagsImagesFolder+'center_of_mass/'
#hashtagsImagesSpreadInTime = hashtagsImagesFolder+'spread_in_time/'
#hashtagsImagesMeanDistanceInTime = hashtagsImagesFolder+'mean_distance/'
hashtagsImagesDisplacementStatsInTime = hashtagsImagesFolder+'displacement_stats/'
hashtagsImagesHashtagsDistributionInLid = hashtagsImagesFolder+'hashtag_dist/%s/'