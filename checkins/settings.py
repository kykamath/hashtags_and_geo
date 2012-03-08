'''
Created on Feb 1, 2012

@author: kykamath
'''

hdfsInputCheckinsFile = 'hdfs:///user/kykamath/geo/checkins/%s/checkins.json'

FOURSQUARE_ID = '4sq'
GOWALLA_ID = 'gowalla'
BRIGHTKITE_ID = 'brightkite'
 
checkinsFolder = '/mnt/chevron/kykamath/data/geo/checkins/%s/'
checkinsAnalysisFolder = '/mnt/chevron/kykamath/data/geo/checkins/analysis/%s/'

checkinsJSONFile = checkinsFolder+'checkins.json'
userToCheckinsMapFile = checkinsAnalysisFolder+'userToCheckinsMap'
lidsToDistributionInSocialNetworksMapFile = checkinsAnalysisFolder + 'lidsToDistributionInSocialNetworksMap'