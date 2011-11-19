'''
Created on Nov 19, 2011

@author: kykamath
'''
import sys, datetime
sys.path.append('../')
from library.file_io import FileIO
from experiments.mr_analysis import MRAnalysis
from library.mrjobwrapper import runMRJob
from settings import hashtagsDistributionInTimeFile, hashtagsDistributionInLatticeFile
import matplotlib.pyplot as plt

def plotHashtagDistributionInTime():
    for h in FileIO.iterateJsonFromFile(hashtagsDistributionInTimeFile):
        if h['t']>3000:
            distribution = dict(h['d'])
#            print unicode(h['h']).encode('utf-8'), h['t']
            ax = plt.subplot(111)
            dataX = sorted(distribution)
            plt.plot_date([datetime.datetime.fromtimestamp(e) for e in dataX], [distribution[x] for x in dataX], '-')
            plt.title('%s (%s)'%(h['h'], h['t']))
            plt.setp(ax.get_xticklabels(), rotation=30, fontsize=10)
            plt.xlim(xmin=datetime.datetime(2011, 2, 1), xmax=datetime.datetime(2011, 11, 30))
            plt.show()

def mr_analysis():
    tempInputFile = 'hdfs:///user/kykamath/geo/twitter/2_11'
    runMRJob(MRAnalysis, hashtagsDistributionInTimeFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
    #runMRJob(MRAnalysis, hashtagsDistributionInLatticeFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
    
if __name__ == '__main__':
    mr_analysis()
    #plotHashtagDistributionInTime()
