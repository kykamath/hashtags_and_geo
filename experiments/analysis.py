'''
Created on Nov 19, 2011

@author: kykamath
'''
import sys, datetime
from library.geo import getHaversineDistance
sys.path.append('../')
from library.file_io import FileIO
from experiments.mr_analysis import MRAnalysis
from library.mrjobwrapper import runMRJob
from settings import hashtagsDistributionInTimeFile, hashtagsDistributionInLatticeFile,\
    hashtagsFile, hashtagsImagesTimeVsDistanceFolder
import matplotlib.pyplot as plt

def plotHashtagDistributionInTime():
    for h in FileIO.iterateJsonFromFile(hashtagsDistributionInTimeFile):
        if h['t']>300:
            distribution = dict(h['d'])
#            print unicode(h['h']).encode('utf-8'), h['t']
            ax = plt.subplot(111)
            dataX = sorted(distribution)
            plt.plot_date([datetime.datetime.fromtimestamp(e) for e in dataX], [distribution[x] for x in dataX])
            plt.title('%s (%s)'%(h['h'], h['t'])    )
            plt.setp(ax.get_xticklabels(), rotation=30, fontsize=10)
            plt.xlim(xmin=datetime.datetime(2011, 2, 1), xmax=datetime.datetime(2011, 11, 30))
            plt.show()

def plotTimeVsDistance():
    i = 1
    for h in FileIO.iterateJsonFromFile(hashtagsFile):
        if h['t']>300:
            occurences = sorted(h['oc'], key=lambda t: t[1])
            initialLattice, initialTime  = occurences[0]
            print i, h['h']; i+=1
            for l, t in occurences[1:]: plt.semilogx(t-initialTime, getHaversineDistance(initialLattice, l), 'o', color='b')
            plt.title('%s (%s)'%(h['h'], h['t']))
            plt.savefig(hashtagsImagesTimeVsDistanceFolder+'%s.png'%h['h'])
            plt.clf()

def mr_analysis():
    tempInputFile = 'hdfs:///user/kykamath/geo/twitter/2_11'
    runMRJob(MRAnalysis, hashtagsFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagsDistributionInTimeFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagsDistributionInLatticeFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
    
if __name__ == '__main__':
#    mr_analysis()
#    plotHashtagDistributionInTime()
    plotTimeVsDistance()
