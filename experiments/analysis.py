'''
Created on Nov 19, 2011

@author: kykamath
'''
import sys, datetime
sys.path.append('../')
from library.geo import getHaversineDistance, getLatticeLid, getLattice,\
    getCenterOfMass
from operator import itemgetter
from experiments.mr_wc import MRWC
from library.file_io import FileIO
from experiments.mr_analysis import MRAnalysis
from library.mrjobwrapper import runMRJob
from settings import hashtagsDistributionInTimeFile, hashtagsDistributionInLatticeFile,\
    hashtagsFile, hashtagsImagesTimeVsDistanceFolder,\
    hashtagsWithoutEndingWindowFile, hashtagsCenterOfMassAnalysisWithoutEndingWindowFile,\
    tempInputFile, inputFolder, hashtagsImagesCenterOfMassFolder
import matplotlib.pyplot as plt
from itertools import combinations
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

def plotCenterOfMassHashtag(timeRange):
    for h in FileIO.iterateJsonFromFile(hashtagsCenterOfMassAnalysisWithoutEndingWindowFile%'%s_%s'%timeRange):
#        print h['h'], h['t'], int(h['ahd'][0][1]), int(h['ahd'][-1][1])
#        print h['h'], h['t'], int(h['com'][0][1][1]), int(h['com'][-1][1][1])
        comData, epData = dict(h['com']), dict(h['ep'])
        dataX = sorted(comData.keys())
        print h['h'], comData[0.1]
        assignedLattice = h['com'][0][1][0]
        initialTime = h['ep'][0][1]
        plt.subplot(211); plt.plot(dataX, [(epData[k]-initialTime)/3600 for k in dataX]); plt.ylabel('Hours')
        plt.title('%s %s (%s)'%(h['h'], assignedLattice, h['t']))
        plt.subplot(212); plt.plot(dataX, [comData[k][1] for k in dataX])
        plt.ylim(ymax=1500)
        plt.savefig('%s/%s_%s.png'%(hashtagsImagesCenterOfMassFolder, h['h'], '%s_%s'%(assignedLattice[0], assignedLattice[1])))
        plt.clf()
#        exit()

def doHashtagCenterOfMassAnalysis(hashtagObject): 
    percentageOfEarlyLattices = [0.01*i for i in range(1, 10)] + [0.1*i for i in range(1, 11)]
    sortedOcc = sorted(hashtagObject['oc'], key=lambda t: t[1])
    llids = [t[0] for t in sortedOcc]
    return {'h':hashtagObject['h'], 't': hashtagObject['t'], 'com': [(p, getCenterOfMass(llids[:int(p*len(llids))], accuracy=0.5, error=True)) for p in percentageOfEarlyLattices]}

def tempAnalysisHashtag(timeRange):
    for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%'%s_%s'%timeRange):
        if h['h']=='occupyboston':
            sortedOcc = sorted(h['oc'], key=lambda t: t[1])
            i=1
            for o in sortedOcc:
                print i, o[0], datetime.datetime.fromtimestamp(o[1]); i+=1
            print doHashtagCenterOfMassAnalysis(h)

def mr_analysis(timeRange):
    def getInputFiles(months): return [inputFolder+str(m) for m in months]
#    runMRJob(MRAnalysis, hashtagsFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagsWithoutEndingWindowFile%'%s_%s'%timeRange, getInputFiles(range(timeRange[0], timeRange[1]+1)), jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagsDistributionInTimeFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagsDistributionInLatticeFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
    runMRJob(MRAnalysis, hashtagsCenterOfMassAnalysisWithoutEndingWindowFile%'%s_%s'%timeRange, getInputFiles(range(timeRange[0], timeRange[1]+1)), jobconf={'mapred.reduce.tasks':300})
    
if __name__ == '__main__':
#    timeRange = (2,5)
    timeRange = (2,11)
    
#    mr_analysis(timeRange)
#    plotHashtagDistributionInTime()
#    plotTimeVsDistance()
#    tempAnalysisHashtag(timeRange)
    plotCenterOfMassHashtag(timeRange)
