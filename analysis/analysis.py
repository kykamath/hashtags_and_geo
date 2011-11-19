'''
Created on Nov 19, 2011

@author: kykamath
'''
from library.mrjobwrapper import runMRJob
from mr_analysis import MRAnalysis
from settings import hashtagsDistributionInTimeFile


tempInputFile = 'hdfs:///user/kykamath/geo/twitter/2_11'
runMRJob(MRAnalysis, hashtagsDistributionInTimeFile, [tempInputFile], jobconf={'mapred.reduce.tasks':90})