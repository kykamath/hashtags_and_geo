'''
Created on Nov 19, 2011

@author: kykamath
'''
import sys, datetime
sys.path.append('../')
from library.geo import getHaversineDistance, getLatticeLid, getLattice,\
    getCenterOfMass, getLocationFromLid
from operator import itemgetter
from experiments.mr_wc import MRWC
from library.file_io import FileIO
from experiments.mr_analysis import MRAnalysis, addHashtagDisplacementsInTime,\
    getMeanDistanceBetweenLids, getMeanDistanceFromSource
from library.mrjobwrapper import runMRJob
from settings import hashtagsDistributionInTimeFile, hashtagsDistributionInLatticeFile,\
    hashtagsFile, hashtagsImagesTimeVsDistanceFolder,\
    hashtagsWithoutEndingWindowFile, \
    tempInputFile, inputFolder, hashtagsImagesCenterOfMassFolder,\
    hashtagsDisplacementStatsFile, hashtagsImagesDisplacementStatsInTime
import matplotlib.pyplot as plt
from itertools import combinations, groupby 
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
        plt.subplot(211); plt.semilogy(dataX, [(epData[k]-initialTime)/3600 for k in dataX]); plt.ylabel('Hours')
        plt.title('%s %s (%s)'%(h['h'], assignedLattice, h['t']))
        plt.subplot(212); plt.plot(dataX, [comData[k][1] for k in dataX])
        plt.ylim(ymax=1500)
        plt.savefig('%s/%s_%s.png'%(hashtagsImagesCenterOfMassFolder, h['h'], '%s_%s'%(assignedLattice[0], assignedLattice[1])))
        plt.clf()
#        exit()

#def doHashtagCenterOfMassAnalysis(hashtagObject): 
#    percentageOfEarlyLattices = [0.01*i for i in range(1, 10)] + [0.1*i for i in range(1, 11)]
#    sortedOcc = sorted(hashtagObject['oc'], key=lambda t: t[1])
#    llids = [t[0] for t in sortedOcc]
#    return {'h':hashtagObject['h'], 't': hashtagObject['t'], 'com': [(p, getCenterOfMass(llids[:int(p*len(llids))], accuracy=0.5, error=True)) for p in percentageOfEarlyLattices]}

#def getSourceLattice(hashtagObject, percentageOfLlidsToConsider):
#    def getMeanDistanceFromSource(source, llids): return np.mean([getHaversineDistance(source, p) for p in llids])
#    sortedOcc = sorted(hashtagObject['oc'], key=lambda t: t[1])[:int(percentageOfLlidsToConsider*len(hashtagObject['oc']))]
#    llids = sorted([t[0] for t in sortedOcc])
#    source = min([(lid, getMeanDistanceFromSource(lid, llids)) for lid in llids], key=lambda t: t[1])
#    return source
#    print getCenterOfMass(llids[:int(percentageOfLlidsToConsider*len(llids))], accuracy=0.5, error=True)

#def addSourceLatticeToHashTagObject(hashtagObject):
#        def getMeanDistanceFromSource(source, llids): return np.mean([getHaversineDistance(source, p) for p in llids])
##        sortedOcc = hashtagObject['oc'][:int(0.01*len(hashtagObject['oc']))]
#        if len(hashtagObject['oc'])>1000: sortedOcc = hashtagObject['oc'][:10]
#        else: sortedOcc = hashtagObject['oc'][:int(0.01*len(hashtagObject['oc']))]
##        llids = sorted([t[0] for t in sortedOcc])
##        uniquellids = [getLocationFromLid(l) for l in set(['%s %s'%(l[0], l[1]) for l in llids])]
##        sourceLlid = min([(lid, getMeanDistanceFromSource(lid, llids)) for lid in uniquellids], key=lambda t: t[1])
##        if sourceLlid[1]>=600: 
#        return max([(lid, len(list(l))) for lid, l in groupby(sorted([t[0] for t in sortedOcc]))], key=lambda t: t[1])
#        else: return sourceLlid


def plotHashtagsDisplacementStats(timeRange):
    MINIMUM_CHECKINS_PER_TIME_INTERVAL = 5
    for h in FileIO.iterateJsonFromFile(hashtagsDisplacementStatsFile%'%s_%s'%timeRange):
#        if h['h'].startswith('occupy'):
        print h['h'] 
        dataX, dataY = [], []
        for x, y in h['sit']:
            if y[0]>=MINIMUM_CHECKINS_PER_TIME_INTERVAL: 
                if y[1]!=0: dataX.append(x), dataY.append(y)
                else: dataX.append(x), dataY.append([y[0], 1])
            else: dataX.append(x), dataY.append([1,1])
        dataX1, dataY1 = [], []
        for x, y in h['mdit']:
            if y[0]>=MINIMUM_CHECKINS_PER_TIME_INTERVAL: 
                if y[1]!=0: dataX1.append(x), dataY1.append(y)
                else: dataX1.append(x), dataY1.append([y[0], 1])
            else: dataX1.append(x), dataY1.append([1,1])
        plt.subplot(311); plt.semilogy([datetime.datetime.fromtimestamp(t) for t in dataX], [y[1] for y in dataY], '-'); plt.xticks([])
        plt.title('%s'%(h['h'])), plt.ylabel('Spread')
        plt.subplot(312); plt.semilogy([datetime.datetime.fromtimestamp(t) for t in dataX1], [y[1] for y in dataY1], '-'); plt.xticks([])
        plt.ylabel('Mean distance')
        ax = plt.subplot(313); plt.semilogy([datetime.datetime.fromtimestamp(t) for t in dataX], [y[0] for y in dataY], '-')
        plt.ylabel('Number of tweets')
        plt.setp(ax.get_xticklabels(), rotation=30, fontsize=10)
#        plt.savefig('%s/%s.png'%(hashtagsImagesDisplacementStatsInTime, h['h']))
        plt.show()
        plt.clf()
#        exit()

def tempAnalysis(timeRange):
    for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%'%s_%s'%timeRange):
        if h['h'].startswith('occupysf'):
#            addHashtagDisplacementsInTime(h, distanceMethod=getMeanDistanceFromSource)
            print h['h'],
            addHashtagDisplacementsInTime(h, distanceMethod=getMeanDistanceBetweenLids)
            print h['sit']
            
def mr_analysis(timeRange):
    def getInputFiles(months): return [inputFolder+str(m) for m in months]
#    runMRJob(MRAnalysis, hashtagsFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagsWithoutEndingWindowFile%'%s_%s'%timeRange, getInputFiles(range(timeRange[0], timeRange[1]+1)), jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagsDistributionInTimeFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagsDistributionInLatticeFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagsCenterOfMassAnalysisWithoutEndingWindowFile%'%s_%s'%timeRange, getInputFiles(range(timeRange[0], timeRange[1]+1)), jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagsSpreadInTimeFile%'%s_%s'%timeRange, getInputFiles(range(timeRange[0], timeRange[1]+1)), jobconf={'mapred.reduce.tasks':300})
    runMRJob(MRAnalysis, hashtagsDisplacementStatsFile%'%s_%s'%timeRange, getInputFiles(range(timeRange[0], timeRange[1]+1)), jobconf={'mapred.reduce.tasks':50})
    
if __name__ == '__main__':
#    timeRange = (2,5)
    timeRange = (2,11)
    
#    mr_analysis(timeRange)
#    plotHashtagDistributionInTime()
#    plotTimeVsDistance()
#    plotHashtagsDisplacementStats(timeRange)
#    plotCenterOfMassHashtag(timeRange)
    tempAnalysis(timeRange)
    