'''
Created on Nov 19, 2011

@author: kykamath
'''
import sys
sys.path.append('../')
from experiments.mr_analysis import MRAnalysis
from library.mrjobwrapper import runMRJob
from settings import hashtagsDistributionInTimeFile


tempInputFile = 'hdfs:///user/kykamath/geo/twitter/2_11'
runMRJob(MRAnalysis, hashtagsDistributionInTimeFile, [tempInputFile], jobconf={'mapred.reduce.tasks':90})