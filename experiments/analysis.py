'''
Created on Nov 19, 2011

@author: kykamath
'''
import sys
sys.path.append('../')
from library.file_io import FileIO
from experiments.mr_analysis import MRAnalysis
from library.mrjobwrapper import runMRJob
from settings import hashtagsDistributionInTimeFile, hashtagsDistributionInLatticeFile


def plotHashtagDistributionInTime():
    for h in FileIO.iterateJsonFromFile(hashtagsDistributionInTimeFile):
        print h

def mr_analysis():
    tempInputFile = 'hdfs:///user/kykamath/geo/twitter/2_11'
    runMRJob(MRAnalysis, hashtagsDistributionInTimeFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
    #runMRJob(MRAnalysis, hashtagsDistributionInLatticeFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
    
plotHashtagDistributionInTime()
