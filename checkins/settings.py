'''
Created on Feb 1, 2012

@author: kykamath
'''

hdfsInputCheckinsFile = 'hdfs:///user/kykamath/geo/checkins/checkins.json'

FOUR_SQUARE = '4sq'
GOWALLA = 'gowalla'
BRIGHTKITE = 'brightkite'
 
checkinsFolder = '/mnt/chevron/kykamath/data/geo/checkins/%s/'
checkinsAnalysisFolder = checkinsFolder+'analysis/'

checkinsJSONFile = checkinsFolder+'checkins.json'
userToCheckinsMapFile = checkinsAnalysisFolder+'userToCheckinsMap'