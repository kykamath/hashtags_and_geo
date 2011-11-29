'''
Created on Nov 28, 2011

@author: kykamath
'''
import sys, os, json, matplotlib, random
sys.path.append('../')
import matplotlib.pyplot as plt
from settings import hashtagsWithoutEndingWindowFile,\
    hashtagsImagesFlowInTimeForWindowOfNOccurrencesFolder
from library.classes import GeneralMethods
from itertools import groupby, combinations
from library.geo import getLocationFromLid, plotPointsOnWorldMap, getLatticeLid
from operator import itemgetter
from multiprocessing import Pool

#GLOBALVAL = 10
#def getFile(i): return '/tmp/00%s.png'%(i+GLOBALVAL)
#def plotFun((i, h)):
#    outputFile = getFile(i)
#    print outputFile
#    plt.plot(h)
#    plt.savefig(outputFile)


ACCURACY = 0.145
TIME_UNIT_IN_SECONDS = 60*60
MIN_OBSERVATIONS_PER_TIME_UNIT = 5
WINDOW_SIZE = 200

def createDirectoryForFile(path):
    dir = path[:path.rfind('/')]
    if not os.path.exists(dir): os.umask(0), os.makedirs('%s'%dir, 0777)
    
def iterateHashtagObjectsFromFile(file):
    numberOfHashtagsPerCall = 100
    hashtagsToYield = []
    for line in open(file): 
        try:
            hashtagsToYield.append(json.loads(line.strip()))
            if len(hashtagsToYield)==numberOfHashtagsPerCall: yield hashtagsToYield; hashtagsToYield=[]
        except: pass
        
def getFileName():
    for i in combinations('abcdefghijklmnopqrstuvwxyz',2): yield ''.join(i)+'.png'

def getValidOccurences(occ, validTimeUnits): return [(p,t) for p,t in occ if GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) in validTimeUnits]
def plotDistributionGraphs(occurences, validTimeUnits, title, startingEpoch=None):
        occurences = getValidOccurences(occurences, validTimeUnits)
        occurancesGroupedByLattice = [(getLocationFromLid(lid.replace('_', ' ')), sorted(zip(*occs)[1])) for lid, occs in groupby(sorted([(getLatticeLid(l, ACCURACY), t) for l, t in occurences], key=itemgetter(0)), key=itemgetter(0))]
        plt.subplot(211)
        pointsForNumberOfOccurances, numberOfOccurancesList = zip(*sorted(occurancesGroupedByLattice, key=lambda t: len(t[1])))
        numberOfOccurancesList = [len(ocs) for ocs in numberOfOccurancesList]
        cm = matplotlib.cm.get_cmap('cool')
        sc = plotPointsOnWorldMap(pointsForNumberOfOccurances, c=numberOfOccurancesList, cmap=cm, lw = 0, alpha=1.0)
        plt.colorbar(sc), plt.title(title), plt.xlabel('Number of mentions')
        
        plt.subplot(212)
        pointsForNumberOfOccurances, occuranceTime = zip(*sorted(occurancesGroupedByLattice, key=lambda t: min(t[1]), reverse=True))
        occuranceTime=[min(t) for t in occuranceTime]
        if not startingEpoch: startingEpoch = occuranceTime[-1]
        occuranceTime=[(t-startingEpoch)/TIME_UNIT_IN_SECONDS for t in occuranceTime]
        cm = matplotlib.cm.get_cmap('autumn')
        sc = plotPointsOnWorldMap(pointsForNumberOfOccurances, c=occuranceTime, cmap=cm, lw = 0, alpha=1.0)
        plt.colorbar(sc), plt.xlabel('Speed of hashtag arrival')
        return startingEpoch

def plotHashtagFlowInTimeForWindowOfNOccurences(hashTagObject):
    def getValidTimeUnits(occ):
        occurranceDistributionInEpochs = [(k[0], len(list(k[1]))) for k in groupby(sorted([GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) for t in zip(*occ)[1]]))]
        return [t[0] for t in occurranceDistributionInEpochs if t[1]>=MIN_OBSERVATIONS_PER_TIME_UNIT]
    currentIndex, previousIndex, startingEpoch = 0, 0, None
    if not os.path.exists(hashtagsImagesFlowInTimeForWindowOfNOccurrencesFolder%hashTagObject['h']):
        validTimeUnits = getValidTimeUnits(hashTagObject['oc'])
        fileNameIterator = getFileName()
        while currentIndex<len(hashTagObject['oc']):
#            try:
                outputFile = hashtagsImagesFlowInTimeForWindowOfNOccurrencesFolder%hashTagObject['h']+fileNameIterator.next(); createDirectoryForFile(outputFile)
                print currentIndex, hashTagObject['h'], outputFile
                currentIndex+=WINDOW_SIZE
                if currentIndex>len(hashTagObject['oc']): currentIndex=len(hashTagObject['oc'])
                occurences = hashTagObject['oc'][previousIndex:currentIndex]
                startingEpoch = plotDistributionGraphs(occurences, validTimeUnits, '%s - Interval (%d - %d) of %d'%(hashTagObject['h'], previousIndex+1, currentIndex, len(hashTagObject['oc'])), startingEpoch)
#                plt.show()
                plt.savefig(outputFile); plt.clf()
                previousIndex=currentIndex
    
#            except: break

timeRange, outputFolder = (2,11), 'world'
#hashtagObjects, counter = [], 0

counter = 0
for hashtagObjects in iterateHashtagObjectsFromFile(hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange)): 
    counter+=len(hashtagObjects); print counter
#    plotHashtagFlowInTimeForWindowOfNOccurences(h)
#    print counter; hashtagObjects.append(h); counter+=1
#print len(hashtagObjects)
#print len(list(iterateHashtagObjectsFromFile('/mnt/chevron/kykamath/data/geo/hashtags/analysis/world/2_11/temp_hashtagsWithoutEndingWindow')))
    po = Pool()
    po.map_async(plotHashtagFlowInTimeForWindowOfNOccurences, hashtagObjects)
    #po.map_async(plotHashtagFlowInTimeForWindowOfNOccurences, iterateHashtagObjectsFromFile('/mnt/chevron/kykamath/data/geo/hashtags/analysis/world/2_11/temp_hashtagsWithoutEndingWindow'))
    po.close(); po.join()
