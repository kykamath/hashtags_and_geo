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
hashtagsFile = hashtagsAnalysisFolder+'%s/%s/hashtags'
hashtagsTrainingDataFile = hashtagsAnalysisFolder+'%s/%s/hashtagsTrainingData'
hashtagsTestDataFile = hashtagsAnalysisFolder+'%s/%s/hashtagsTestData'
hashtagsWithoutEndingWindowFile = hashtagsAnalysisFolder+'%s/%s/hashtagsWithoutEndingWindow'
hashtagsWithKnownSourceFile = hashtagsAnalysisFolder+'%s/%s/hashtagsWithKnownSource'
hashtagsLatticeGraphFile = hashtagsAnalysisFolder+'%s/%s/latticeGraph'

hashtagsWithoutEndingWindowAndOcccurencesFilteredByDistributionInTimeUnitsFile = hashtagsAnalysisFolder+'%s/%s/hashtagsWithoutEndingWindowAndOcccurencesFilteredByDistributionInTimeUnits'
hashtagsBoundarySpecificStatsFile = hashtagsAnalysisFolder+'%s/%s/hashtagsBoundarySpecificStats'
hashtagsDisplacementStatsFile = hashtagsAnalysisFolder+'%s/%s/hashtagsDisplacementStats'
hashtagsDistributionInTimeFile = hashtagsAnalysisFolder+'hashtagsDistributionInTime'
hashtagsDistributionInLatticeFile = hashtagsAnalysisFolder+'hashtagsDistributionInLattice'
hashtagsAnalayzeLocalityIndexAtKFile = hashtagsAnalysisFolder+'%s/%s/hashtagsAnalayzeLocalityIndexAtK'
hashtagWithGuranteedSourceFile = hashtagsAnalysisFolder+'%s/%s/hashtagWithGuranteedSource'
hashtagSharingProbabilityGraphFile = hashtagsAnalysisFolder+'%s/%s/hashtagSharingProbabilityGraph'
hashtagSharingProbabilityGraphWithTemporalClosenessFile = hashtagsAnalysisFolder+'%s/%s/hashtagSharingProbabilityGraphWithTemporalCloseness'
hashtagLocationTemporalClosenessGraphFile = hashtagsAnalysisFolder+'%s/%s/locationTemporalClosenessGraph'
hashtagLocationInAndOutTemporalClosenessGraphFile = hashtagsAnalysisFolder+'%s/%s/locationInAndOutTemporalClosenessGraph'

# Images
hashtagsImagesFolder = '/data/geo/hashtags/images/%s/'
hashtagsImagesFlowInTimeForFirstNLocationsFolder = hashtagsImagesFolder+'fit_first_n_locations/%s/'
hashtagsImagesFlowInTimeForFirstNOccurrencesFolder = hashtagsImagesFolder+'fit_first_n_occ/%s/'
hashtagsImagesFlowInTimeForWindowOfNOccurrencesFolder = hashtagsImagesFolder+'fit_window_of_n_occ/%s/'
hashtagsImagesFlowInTimeForWindowOfNLocationsFolder = hashtagsImagesFolder+'fit_window_of_n_loc/%s/'
hashtagsImagesLocationClosenessFolder = hashtagsImagesFolder + 'location_closeness/'
hashtagsImagesLocationInfluencersFolder = hashtagsImagesFolder + 'location_influencers/'
hashtagsImagesTimeSeriesAnalysisFolder = hashtagsImagesFolder + 'time_series/'
hashtagsImagesHashtagsClassFolder = hashtagsImagesFolder + 'hashtags_class/'
hashtagsImagesFirstActiveTimeSeriesAnalysisFolder = hashtagsImagesFolder + 'first_active_time_series/'
hashtagsImagesGraphAnalysisFolder = hashtagsImagesFolder + 'graph_analysis/'
hashtagsImagesHastagEvolutionFolder = hashtagsImagesFolder + 'hashtag_evolution/'


# Classifiers
hashtagsClassifiersFolder = hashtagsAnalysisFolder+'/classifiers/%s/%s/'

#Models
hashtagsModelsFolder = hashtagsAnalysisFolder+'/models/%s/%s/'