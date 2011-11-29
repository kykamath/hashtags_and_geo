'''
Created on Nov 28, 2011

@author: kykamath
'''
import sys, os, json, matplotlib, random
sys.path.append('../')
import matplotlib.pyplot as plt
from settings import hashtagsWithoutEndingWindowFile,\
    hashtagsImagesFlowInTimeForWindowOfNOccurrencesFolder,\
    hashtagsImagesFlowInTimeForFirstNLocationsFolder,\
    hashtagsImagesFlowInTimeForWindowOfNLocationsFolder
from library.classes import GeneralMethods
from itertools import groupby, combinations
from library.geo import getLocationFromLid, plotPointsOnWorldMap, getLatticeLid
from operator import itemgetter
from multiprocessing import Pool
from collections import defaultdict

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
OCCURENCE_WINDOW_SIZE = 200
LOCATION_WINDOW_SIZE = 20

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

def getValidTimeUnits(occ):
    occurranceDistributionInEpochs = [(k[0], len(list(k[1]))) for k in groupby(sorted([GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) for t in zip(*occ)[1]]))]
    return [t[0] for t in occurranceDistributionInEpochs if t[1]>=MIN_OBSERVATIONS_PER_TIME_UNIT]

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
    currentIndex, previousIndex, startingEpoch = 0, 0, None
    if not os.path.exists(hashtagsImagesFlowInTimeForWindowOfNOccurrencesFolder%hashTagObject['h']):
        validTimeUnits = getValidTimeUnits(hashTagObject['oc'])
        fileNameIterator = getFileName()
        while currentIndex<len(hashTagObject['oc']):
            try:
                outputFile = hashtagsImagesFlowInTimeForWindowOfNOccurrencesFolder%hashTagObject['h']+fileNameIterator.next(); createDirectoryForFile(outputFile)
                print currentIndex, hashTagObject['h'], outputFile
                currentIndex+=OCCURENCE_WINDOW_SIZE
                if currentIndex>len(hashTagObject['oc']): currentIndex=len(hashTagObject['oc'])
                occurences = hashTagObject['oc'][previousIndex:currentIndex]
                startingEpoch = plotDistributionGraphs(occurences, validTimeUnits, '%s - Interval (%d - %d) of %d'%(hashTagObject['h'], previousIndex+1, currentIndex, len(hashTagObject['oc'])), startingEpoch)
#                plt.show()
                plt.savefig(outputFile); plt.clf()
                previousIndex=currentIndex
            except: break

def plotHashtagFlowInTimeForWindowOfNLocations(hashTagObject):
    currentIndex, previousIndex, startingEpoch = 0, 0, None
    if not os.path.exists(hashtagsImagesFlowInTimeForWindowOfNLocationsFolder%hashTagObject['h']):
        validTimeUnits, latticesToOccranceMap = getValidTimeUnits(hashTagObject['oc']), defaultdict(list)
        fileNameIterator = getFileName()
        for l, t in hashTagObject['oc']: latticesToOccranceMap[getLatticeLid(l, ACCURACY)].append((l,t))
        for k in latticesToOccranceMap.keys()[:]: 
            validOccurences = getValidOccurences(latticesToOccranceMap[k], validTimeUnits)
            if validOccurences: latticesToOccranceMap[k] =  validOccurences
            else: del latticesToOccranceMap[k]
        latticesSortedByTime = sorted([(k, min(zip(*v)[1])) for k, v in latticesToOccranceMap.iteritems()], key=itemgetter(1))
        while currentIndex<len(latticesSortedByTime):
            try:
                outputFile = hashtagsImagesFlowInTimeForWindowOfNLocationsFolder%hashTagObject['h']+fileNameIterator.next(); createDirectoryForFile(outputFile)
                print currentIndex, hashTagObject['h'], outputFile
                currentIndex+=LOCATION_WINDOW_SIZE
                if currentIndex>len(latticesSortedByTime): currentIndex=len(latticesSortedByTime)
                occurences = []
                for l in latticesSortedByTime[previousIndex:currentIndex]: occurences+=latticesToOccranceMap[l[0]]
                startingEpoch = plotDistributionGraphs(occurences, validTimeUnits, '%s - Interval (%d - %d) of %d'%(hashTagObject['h'], previousIndex+1, currentIndex, len(latticesSortedByTime)), startingEpoch)
#                plt.show()
                plt.savefig(outputFile); plt.clf()
                previousIndex=currentIndex
            except: break

timeRange, outputFolder = (2,11), 'world'
counter = 0
'''
ls -al /data/geo/hashtags/images/fit_window_of_n_occ/ | wc -l
'''
for hashtagObjects in iterateHashtagObjectsFromFile(hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange)): 
    counter+=len(hashtagObjects); print counter
    po = Pool()
    po.map_async(plotHashtagFlowInTimeForWindowOfNLocations, hashtagObjects)
    po.close(); po.join()

#outputFolder = '/'
#for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange)):
#    plotHashtagFlowInTimeForWindowOfNLocations(h)
