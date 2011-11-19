'''
Created on Nov 19, 2011

@author: kykamath
'''

from library.mrjobwrapper import runMRJob
from analysis.mr_analysis import MRAnalysis

tempInputFile = 'hdfs:///user/kykamath/geo/twitter/2_5'
runMRJob(MRAnalysis, 'temp_file', [tempInputFile], jobconf={'mapred.reduce.tasks':90})