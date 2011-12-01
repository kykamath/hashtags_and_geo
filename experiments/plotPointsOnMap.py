'''
Created on Nov 28, 2011

@author: kykamath
'''
import sys, os, json, matplotlib, random, datetime
from library.file_io import FileIO
from experiments.mr_analysis import TIME_UNIT_IN_SECONDS,\
    getOccurranceDistributionInEpochs
sys.path.append('../')
import matplotlib.pyplot as plt
from settings import hashtagsWithoutEndingWindowFile,\
    hashtagsImagesFlowInTimeForWindowOfNOccurrencesFolder,\
    hashtagsImagesFlowInTimeForFirstNLocationsFolder,\
    hashtagsImagesFlowInTimeForWindowOfNLocationsFolder,\
    hashtagsImagesNodeFolder, hashtagSharingProbabilityGraphFile,\
    hashtagsImagesTimeSeriesAnalysisFolder,\
    hashtagsWithoutEndingWindowAndOcccurencesFilteredByDistributionInTimeUnitsFile
from mpl_toolkits.axes_grid1 import make_axes_locatable
from library.classes import GeneralMethods
from itertools import groupby, combinations
from library.geo import getLocationFromLid, plotPointsOnWorldMap, getLatticeLid
from operator import itemgetter
from multiprocessing import Pool
from collections import defaultdict

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

def getOccurencesFilteredByDistributionInTimeUnits(occ, validTimeUnits): return [(p,t) for p,t in occ if GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) in validTimeUnits]
def plotDistributionGraphs(occurences, validTimeUnits, title, startingEpoch=None):
        occurences = getOccurencesFilteredByDistributionInTimeUnits(occurences, validTimeUnits)
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
            validOccurences = getOccurencesFilteredByDistributionInTimeUnits(latticesToOccranceMap[k], validTimeUnits)
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
            
def plotNodeObject(nodeObject):
    ax = plt.subplot(111)
    cm = matplotlib.cm.get_cmap('cool')
    point = getLocationFromLid(nodeObject['id'].replace('_', ' '))
    outputFile = hashtagsImagesNodeFolder+'%s.png'%getLatticeLid([point[1], point[0]], ACCURACY); createDirectoryForFile(outputFile)
    if not os.path.exists(outputFile):
        print outputFile
        points, colors = zip(*sorted([(getLocationFromLid(k.replace('_', ' ')), v)for k, v in nodeObject['links'].iteritems()], key=itemgetter(1)))
        sc = plotPointsOnWorldMap(points, c=colors, cmap=cm, lw=0)
        plotPointsOnWorldMap([getLocationFromLid(nodeObject['id'].replace('_', ' '))], c='k', s=20, lw=0)
        plt.xlabel('Measure of closeness'), plt.title(nodeObject['id'].replace('_', ' '))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc, cax=cax)
    #    plt.show()
        plt.savefig(outputFile); plt.clf()
        
def plotTimeSeriesRealData(hashtagObject, id='real_data'):
#    MIN_OBSERVATIONS_PER_TIME_UNIT = 2
    ax = plt.subplot(111)
    outputFile = hashtagsImagesTimeSeriesAnalysisFolder%(id, hashtagObject['h'])+'%s.png'%id; createDirectoryForFile(outputFile)
    print unicode(outputFile).encode('utf-8')
    occurranceDistributionInEpochs = sorted([(k[0], len(list(k[1]))) 
                                                 for k in groupby(sorted(
                                                                          [GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS) for t in zip(*hashtagObject['oc'])[1]])
                                                                  )
                                             ], key=itemgetter(0))
#    occurranceDistributionInEpochs = filter(lambda t: t[1]>=MIN_OBSERVATIONS_PER_TIME_UNIT, occurranceDistributionInEpochs)
    dataX, dataY = zip(*occurranceDistributionInEpochs)
    plt.plot_date(map(datetime.datetime.fromtimestamp, dataX), dataY, '-')
    plt.setp(ax.get_xticklabels(), rotation=30, fontsize=10)
    plt.title('%s - %s'%(hashtagObject['h'], id))
#    plt.show()
    plt.savefig(outputFile); plt.clf()

#    exit()
   

timeRange, outputFolder = (2,11), 'world'
counter = 0
'''
ls -al /data/geo/hashtags/images/fit_window_of_n_occ/ | wc -l
'''
#for hashtagObjects in iterateHashtagObjectsFromFile(hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange)): 
#for object in iterateHashtagObjectsFromFile(hashtagsWithoutEndingWindowFile%(outputFolder, '%s_%s'%timeRange)): 
#    counter+=len(object); print counter
#    po = Pool()
#    po.map_async(plotTimeSeries, object)
#    po.close(); po.join()

#for object in FileIO.iterateJsonFromFile(hashtagSharingProbabilityGraphFile%(outputFolder, '%s_%s'%timeRange)):
#for object in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowAndOcccurencesFilteredByDistributionInTimeUnitsFile%(outputFolder, '%s_%s'%timeRange)):
#    print counter; counter+=1
#    plotNodeObject(object)
