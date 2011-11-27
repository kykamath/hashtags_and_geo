'''
Created on Nov 19, 2011

@author: kykamath
'''
import sys, datetime
from library.classes import GeneralMethods
sys.path.append('../')
from library.geo import getHaversineDistance, getLatticeLid, getLattice,\
    getCenterOfMass, getLocationFromLid, plotPointsOnUSMap
from operator import itemgetter
from experiments.mr_wc import MRWC
from library.file_io import FileIO
from experiments.mr_analysis import MRAnalysis, addHashtagDisplacementsInTime,\
    getMeanDistanceBetweenLids, getMeanDistanceFromSource, getLocalityIndexAtK,\
    addSourceLatticeToHashTagObject, addHashtagLocalityIndexInTime,\
    HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS
from library.mrjobwrapper import runMRJob
from settings import hashtagsDistributionInTimeFile, hashtagsDistributionInLatticeFile,\
    hashtagsFile, hashtagsImagesTimeVsDistanceFolder,\
    hashtagsWithoutEndingWindowFile, \
    tempInputFile, inputFolder, hashtagsImagesCenterOfMassFolder,\
    hashtagsDisplacementStatsFile, hashtagsImagesDisplacementStatsInTime,\
    hashtagsImagesHashtagsDistributionInLid,\
    hashtagsAnalayzeLocalityIndexAtKFile, hashtagWithGuranteedSourceFile,\
    hashtagsBoundarySpecificStatsFile
import matplotlib.pyplot as plt
from itertools import combinations, groupby 
import numpy as np
from collections import defaultdict

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
            
class HashtagObject:
    def __init__(self, hashtagDict): self.dict = hashtagDict
    def getLocalLattice(self): return self.dict['liAtVaryingK'][0][1]
    @staticmethod
    def iterateHashtagObjects(timeRange, file=hashtagsAnalayzeLocalityIndexAtKFile, hastagsList = None):
        for h in FileIO.iterateJsonFromFile(file%(outputFolder, '%s_%s'%timeRange)): 
            if hastagsList==None or h['h'] in hastagsList: yield HashtagObject(h)

class HashtagClass:
    hashtagsCSV = '../data/hashtags.csv'
    @staticmethod
    def createCSV(timeRange):
        i=1
        for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange)):
            FileIO.writeToFile(h['h']+', ', HashtagClass.hashtagsCSV)
    @staticmethod
    def iterateHashtagsWithClass():
        for line in FileIO.iterateLinesFromFile(HashtagClass.hashtagsCSV): 
            h, cls = line.strip().split(', ')   
            yield h.strip(), cls.strip()
    @staticmethod
    def getHashtagClasses():
        hashtags = sorted(HashtagClass.iterateHashtagsWithClass(), key=itemgetter(1))
        return dict((cls, list(h[0] for h in hashtags)) for cls, hashtags in groupby(hashtags, key=itemgetter(1)))


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
    for h in FileIO.iterateJsonFromFile(hashtagsCenterOfMassAnalysisWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange)):
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

#def getLatticeThatGivesMinimumLocalityIndexAtKForAccuracy(occurances, kValue, accuracy):
#    occurancesDistributionInHigherLattice, distanceMatrix = defaultdict(list), defaultdict(dict)
#    for oc in occurances: occurancesDistributionInHigherLattice[getLatticeLid(oc, accuracy)].append(oc)
#    higherLattices = sorted(occurancesDistributionInHigherLattice.iteritems(), key=lambda t: len(t[1]), reverse=True)
#    for hl1, hl2 in combinations(occurancesDistributionInHigherLattice, 2): distanceMatrix[hl1][hl2] = distanceMatrix[hl2][hl1] = getHaversineDistance(getLocationFromLid(hl1.replace('_', ' ')), getLocationFromLid(hl2.replace('_', ' ')))
#    for k,v in distanceMatrix.iteritems(): distanceMatrix[k] = sorted(v.iteritems(), key=itemgetter(1))
#    occurancesToReturn = []
#    currentHigherLatticeSet, totalOccurances = {'distance': ()}, float(len(occurances))
#    for hl, occs  in higherLattices: 
#        higherLatticeSet = {'distance': 0, 'observedOccurances': len(occs), 'lattices': [hl], 'sourceLattice': hl}
#        while currentHigherLatticeSet['distance']>higherLatticeSet['distance'] and higherLatticeSet['observedOccurances']/totalOccurances<kValue:
#            (l, d) = distanceMatrix[hl][0]; 
#            distanceMatrix[hl]=distanceMatrix[hl][1:]
#            higherLatticeSet['distance']+=d
#            higherLatticeSet['lattices'].append(l)
#            higherLatticeSet['observedOccurances']+=len(occurancesDistributionInHigherLattice[l])
#        if currentHigherLatticeSet==None or currentHigherLatticeSet['distance']>higherLatticeSet['distance']: currentHigherLatticeSet=higherLatticeSet
#    
#    for l in currentHigherLatticeSet['lattices']: occurancesToReturn+=occurancesDistributionInHigherLattice[l]
##    return {'distance': currentHigherLatticeSet['distance'], 'occurances': occurancesToReturn, 'sourceLattice': getLocationFromLid(currentHigherLatticeSet['sourceLattice'].replace('_', ' '))}
#    return {'occurances': occurancesToReturn, 'sourceLattice': getLocationFromLid(currentHigherLatticeSet['sourceLattice'].replace('_', ' '))}


def tempGetLocality(timeRange):
#    K_FOR_LOCALITY_INDEX = 0.5
    for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange)):
        if h['h'].startswith('occupy') or h['h']=='ows':
##            print h['h'], getLocalityIndexAtK(zip(*h['oc'])[0], K_FOR_LOCALITY_INDEX)
#            occurances = zip(*h['oc'])[0]
#            print h['h'], [(k, getLocalityIndexAtK(occurances, k)) for k in [0.5+0.05*i for i in range(11)]]
            addSourceLatticeToHashTagObject(h)
            print h['h'], h['src'], int(h['t']*0.01), h['src'][1]/(h['t']*0.01)
                            
def plotOnUsMap(timeRange):
    '''convert -delay 60 -quality 95 * movie.gif
    '''
    HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS = 60*60
    MINIMUM_CHECKINS_PER_TIME_INTERVAL = 10
    ACCURACY = 0.5
    SOURCE_COLOR = 'r'; OTHER_POINTS_COLOR = 'b'
    for h in FileIO.iterateJsonFromFile(hashtagsDisplacementStatsFile%(outputFolder, '%s_%s'%timeRange)):
        if h['h'].startswith('occupy'):
            occurencesDistribution = defaultdict(list)
            source = getLatticeLid(h['src'][0], ACCURACY)
            for oc in h['oc']: occurencesDistribution[GeneralMethods.approximateEpoch(oc[1], HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS)].append(oc)
            for ep in sorted(occurencesDistribution):
                ep-=18000
                v = occurencesDistribution[ep]
#                print h['h'], datetime.datetime.fromtimestamp(ep), len(v)
                if len(v) >= MINIMUM_CHECKINS_PER_TIME_INTERVAL:
                    llids = [getLatticeLid(p, ACCURACY) for p in zip(*v)[0]]
                    points, pointSizes = zip(*[(k, len(list(l)))for k,l in groupby(sorted(llids))])
                    pointSizes, pointsColor, pointsLabel = [p*100 for p in pointSizes], [], []
                    for p in points: 
                        if p==source: pointsColor.append(SOURCE_COLOR)
                        else: pointsColor.append(OTHER_POINTS_COLOR)
                    points = [getLocationFromLid(p.replace('_', ' ')) for p in points]
                    plotPointsOnUSMap(points, pointSize=pointSizes, pointColor=pointsColor)
                    imageFile = hashtagsImagesHashtagsDistributionInLid%h['h']+'%s.png'%ep
                    print imageFile
                    plt.title('%s - %s'%(h['h'], str(datetime.datetime.fromtimestamp(ep))))
                    FileIO.createDirectoryForFile(imageFile)
                    plt.savefig(imageFile)
                    plt.clf()
                    exit()
        exit()

def plotHashtagsDisplacementStats(timeRange):
    MINIMUM_CHECKINS_PER_TIME_INTERVAL = 0
    for h in FileIO.iterateJsonFromFile(hashtagsDisplacementStatsFile%(outputFolder, '%s_%s'%timeRange)):
#        if h['h'].startswith('occupy'):
        print h['h'] 
        dataX, dataY = [], []
        for x, y in sorted(h['sit']):
            if y[0]>=MINIMUM_CHECKINS_PER_TIME_INTERVAL: 
                if y[1]!=0: dataX.append(x), dataY.append(y)
                else: dataX.append(x), dataY.append([y[0], 1])
            else: dataX.append(x), dataY.append([1,1])
        dataX1, dataY1 = [], []
        for x, y in sorted(h['liInTime']):
            if y[0]>=MINIMUM_CHECKINS_PER_TIME_INTERVAL: 
                if y[1]!=0: dataX1.append(x), dataY1.append(y)
                else: dataX1.append(x), dataY1.append([y[0], 1])
            else: dataX1.append(x), dataY1.append([1,1])
        plt.subplot(311); plt.semilogy([datetime.datetime.fromtimestamp(t) for t in dataX], [y[1] for y in dataY], '-'); plt.xticks([])
        plt.title('%s'%(h['h'])), plt.ylabel('Spread'), plt.ylim(ymin=4, ymax=10**4)
        plt.subplot(312); 
        plt.plot([datetime.datetime.fromtimestamp(t) for t in dataX1], [y[1] for y in dataY1], '-'); plt.xticks([])
        plt.ylabel('LI'), #plt.ylim(ymin=4, ymax=10**4)
        ax = plt.subplot(313); plt.semilogy([datetime.datetime.fromtimestamp(t) for t in dataX], [y[0] for y in dataY], '-')
        plt.ylabel('Number of tweets')
        plt.setp(ax.get_xticklabels(), rotation=30, fontsize=10)
#        plt.savefig('%s/%s.png'%(hashtagsImagesDisplacementStatsInTime, h['h']))
        plt.show()
        plt.clf()
#        exit()

def analayzeLocalityIndexAtK(timeRange):
    imagesFolder = '/tmp/images/'
    occupyList = ['occupywallst', 'occupyoakland', 'occupydc', 'occupysf']
    FileIO.createDirectoryForFile(imagesFolder+'dsf')
    for h in FileIO.iterateJsonFromFile(hashtagsAnalayzeLocalityIndexAtKFile%(outputFolder, '%s_%s'%timeRange)):
#        if h['h'].startswith('occupy') or h['h']=='ows':
        if h['h'] in occupyList:
            print h['h']#, h['liAtVaryingK']
            dataX, dataY = zip(*[(k, v[0])for k, v in h['liAtVaryingK']])
            plt.plot(dataX, dataY, label=h['h'])
#    plt.title(h['h'])
    plt.ylim(ymax=4000)
    plt.legend(loc=2)
    plt.show()
            
def tempAnalysis(timeRange):
    for h in FileIO.iterateJsonFromFile(hashtagsAnalayzeLocalityIndexAtKFile%(outputFolder, '%s_%s'%timeRange)):
        if h['liAtVaryingK'][0][1][0] > 0:
            print h['h']
            addHashtagLocalityIndexInTime(h)
            exit()
#        if h['h'].startswith('occupysf'):
##            addHashtagDisplacementsInTime(h, distanceMethod=getMeanDistanceFromSource)
#            print h['h'],
#            addHashtagDisplacementsInTime(h, distanceMethod=getMeanDistanceBetweenLids)
#            print h['sit']

def plotChangeLIOnUSMap(timeRange):
    hashtagClasses = HashtagClass.getHashtagClasses()
#    classesToPlot = ['technology', 'tv', 'movie', 'sports', 'occupy', 'events', 'republican_debates', 'song_game_releases']
    
    folderName = '/tmp/images/%s/%s.png'
    classesToPlot = ['occupy']
    for cls in classesToPlot:
        for h in FileIO.iterateJsonFromFile(hashtagsDisplacementStatsFile%(outputFolder, '%s_%s'%timeRange)):
            if h['h'] in hashtagClasses[cls]:
                occurencesDistribution = defaultdict(list)
                for oc in h['oc']: occurencesDistribution[GeneralMethods.approximateEpoch(oc[1], HASHTAG_SPREAD_ANALYSIS_WINDOW_IN_SECONDS)].append(oc)
                dataX = sorted(occurencesDistribution)
                dataY = [len(occurencesDistribution[x]) for x in dataX]
                dataX = [datetime.datetime.fromtimestamp(x) for x in dataX]
                for t, l in sorted(h['liInTime'], key=itemgetter(0)):
                    print h['h'], t, l
                    plt.subplot(211), plt.title(h['h'] + ' ' + str(datetime.datetime.fromtimestamp(t)))
                    plotPointsOnUSMap([l[1]], pointSize=[100+l[0]], pointColor=['r'])
                    ax = plt.subplot(212)
                    plt.semilogy(dataX, dataY, '-')
                    plt.plot(datetime.datetime.fromtimestamp(t), len(occurencesDistribution[t]), 'o')
                    plt.setp(ax.get_xticklabels(), rotation=30, fontsize=10)
#                plotPointsOnUSMap([t[1] for t in zip(*h['liInTime'])[1]])
#                    plt.show()
                    FileIO.createDirectoryForFile(folderName%(h['h'], t))
                    plt.savefig(folderName%(h['h'], t))
                    plt.clf()
#        exit()

def mr_analysis(timeRange, outputFolder):
    def getInputFiles(months, folderType='/'): return [inputFolder+folderType+'/'+str(m) for m in months]
#    runMRJob(MRAnalysis, hashtagsFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1)), jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagsDistributionInTimeFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagsDistributionInLatticeFile, [tempInputFile], jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagsCenterOfMassAnalysisWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1)), jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagsSpreadInTimeFile%(outputFolder, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1)), jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagsDisplacementStatsFile%(outputFolder, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), outputFolder), jobconf={'mapred.reduce.tasks':90})
#    runMRJob(MRAnalysis, hashtagsAnalayzeLocalityIndexAtKFile%(outputFolder, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1)), jobconf={'mapred.reduce.tasks':300})
#    runMRJob(MRAnalysis, hashtagWithGuranteedSourceFile%(outputFolder, '%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1)), jobconf={'mapred.reduce.tasks':300})
    runMRJob(MRAnalysis, hashtagsBoundarySpecificStatsFile%(outputFolder,'%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), outputFolder), jobconf={'mapred.reduce.tasks':90})

if __name__ == '__main__':
#    timeRange = (2,5)
    timeRange = (2,11)
    
#    outputFolder = 'us'
    outputFolder = 'world'
    
    mr_analysis(timeRange, outputFolder)
#    plotHashtagDistributionInTime()
#    plotTimeVsDistance()
#    plotChangeLIOnUSMap(timeRange)
#    plotHashtagsDisplacementStats(timeRange)
#    plotCenterOfMassHashtag(timeRange)
#    tempAnalysis(timeRange)
#    plotOnUsMap(timeRange)
#    tempGetLocality(timeRange)
#    analayzeLocalityIndexAtK(timeRange)
