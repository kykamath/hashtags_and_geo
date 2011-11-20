'''
Created on Nov 19, 2011

@author: kykamath
'''
import sys, datetime
sys.path.append('../')
from library.geo import getHaversineDistance, getLatticeLid
from experiments.mr_wc import MRWC
from library.file_io import FileIO
from experiments.mr_analysis import MRAnalysis
from library.mrjobwrapper import runMRJob
from settings import hashtagsDistributionInTimeFile, hashtagsDistributionInLatticeFile,\
    hashtagsFile, hashtagsImagesTimeVsDistanceFolder,\
    hashtagsWithoutEndingWindowFile
import matplotlib.pyplot as plt
import numpy as np

def getModifiedLatticeLid(point, accuracy=0.0075):
    return '%0.1f_%0.1f'%(int(point[0]/accuracy)*accuracy, int(point[1]/accuracy)*accuracy)

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
#    def printLattice(lattice): return '%0.1f '
    i = 1
    for h in FileIO.iterateJsonFromFile(hashtagsFile):
        if h['t']>300:
            occurrenceTimes = [t[1] for t in h['oc']] 
            mean, std = np.mean(occurrenceTimes), np.std(occurrenceTimes)
#            print mean, std
            window=1
            lowerTimeBoundary, upperTimeBoundary = int(mean-window*std), int(mean+window*std)
            occurences = sorted([t for t in h['oc'] if  t[1]>=lowerTimeBoundary and t[1]<=upperTimeBoundary], key=lambda t: t[1])
#            print len(h['oc']), len(occurences)
            initialLattice, initialTime  = occurences[0]
            print i, h['h'], len(h['oc']), int(len(h['oc'])*0.01) #[getModifiedLatticeLid(l[0],accuracy=1.45).replace('_', ',') for l in occurences[:5]]  ; i+=1;
#            for l,t in occurences: plt.semilogx(t-initialTime, getHaversineDistance(initialLattice, l), 'o', color='b')
            
#            for l, t in occurences[1:]: plt.plot(t-mean, getHaversineDistance(initialLattice, l), 'o', color='b')
#            plt.title('%s (%s)'%(h['h'], h['t']))
#            plt.savefig(hashtagsImagesTimeVsDistanceFolder+'%s.png'%h['h'])
#            plt.clf()

def mr_analysis():
    tempInputFile = 'hdfs:///user/kykamath/geo/twitter/2_11'
#    runMRJob(MRAnalysis, hashtagsFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
    runMRJob(MRAnalysis, hashtagsWithoutEndingWindowFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagsDistributionInTimeFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagsDistributionInLatticeFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
    
if __name__ == '__main__':
    mr_analysis()
#    plotHashtagDistributionInTime()
#    plotTimeVsDistance()

