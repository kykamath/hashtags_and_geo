'''
Created on Dec 7, 2011

@author: kykamath
'''
import sys
sys.path.append('../')
from itertools import groupby, combinations
from operator import itemgetter
from library.file_io import FileIO
from settings import hashtagsWithKnownSourceFile,\
    hashtagsImagesHastagEvolutionFolder, hashtagsFile
from experiments.mr_area_analysis import getOccuranesInHighestActiveRegion,\
    LATTICE_ACCURACY, getSourceLattice,\
    MIN_OCCURRENCES_TO_DETERMINE_SOURCE_LATTICE, TIME_UNIT_IN_SECONDS,\
    getOccurranceDistributionInEpochs, HashtagsClassifier
from library.geo import getLocationFromLid, getLatticeLid, plotPointsOnWorldMap
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os


def getFileName():
    for i in combinations('abcdefghijklmnopqrstuvwxyz',8): yield ''.join(i)+'.png'

timeRange = (2,11)
outputFolder = 'world'
#counter=1
#WINDOW = 20
#for hashtagObject in FileIO.iterateJsonFromFile(hashtagsWithKnownSourceFile%(outputFolder,'%s_%s'%timeRange)):
#    print counter; counter+=1
#    hashtagsFolder = hashtagsImagesHastagEvolutionFolder%outputFolder+hashtagObject['h']
#    occuranesInHighestActiveRegion, isFirstActiveRegion = getOccuranesInHighestActiveRegion(hashtagObject, True)
#    source, count = getSourceLattice(occuranesInHighestActiveRegion)
##    print count, isFirstActiveRegion
##    assert count>=MIN_OCCURRENCES_TO_DETERMINE_SOURCE_LATTICE and isFirstActiveRegion==True
#    numberOfOccurences = len(occuranesInHighestActiveRegion)
#    if numberOfOccurences>1000 and not os.path.exists(hashtagsFolder):
#        currentIndex = WINDOW
#        fileNameIterator = getFileName()
#        sourceTime = None
#        while currentIndex<=numberOfOccurences:
#            occurances = occuranesInHighestActiveRegion[:currentIndex]
#            if not sourceTime: sourceTime=occurances[0][1]
#            td = float(occurances[-1][1]-sourceTime)
#    #        print len(hashtagObject['oc'][:currentIndex]), numberOfOccurences 
#            outputFile = hashtagsFolder+'/'+fileNameIterator.next(); FileIO.createDirectoryForFile(outputFile)
#            print outputFile
#            occurancesGroupedByLattice = [(getLocationFromLid(lid.replace('_', ' ')), len(sorted(zip(*occs)[1]))) for lid, occs in groupby(sorted([(getLatticeLid(l, LATTICE_ACCURACY), t) for l, t in occurances], key=itemgetter(0)), key=itemgetter(0))]
#            points, colors = zip(*occurancesGroupedByLattice)
#            ax = plt.subplot(111)
#            cm = matplotlib.cm.get_cmap('cool')
#            sc = plotPointsOnWorldMap(points, c=colors, cmap=cm, lw = 0)
#            plotPointsOnWorldMap([source], c='g', alpha=0.5, s=50, lw=1)
#            plt.title('%s (%s of %s) (%0.3f - %d)'%(hashtagObject['h'], len(occurances), numberOfOccurences, td/TIME_UNIT_IN_SECONDS, td))
#            divider = make_axes_locatable(ax)
#            cax = divider.append_axes("right", size="5%", pad=0.05)
#            plt.colorbar(sc, cax=cax)
#            plt.show()
##            plt.savefig(outputFile);plt.clf()
#            currentIndex+=WINDOW
#    exit()

#for hashtagObject in FileIO.iterateJsonFromFile('/mnt/chevron/kykamath/data/geo/hashtags/analysis/new_world/2_11/hashtags'):
#    if hashtagObject['h']=='ripamywinehouse':
#        print unicode(hashtagObject['h']).encode('utf-8')
#        occsDistributionInTimeUnits = getOccurranceDistributionInEpochs(getOccuranesInHighestActiveRegion(hashtagObject), timeUnit=60*60, fillInGaps=True, occurancesCount=False)
#        totalOccurances = []
#        for t, occs in occsDistributionInTimeUnits.iteritems():
#            totalOccurances+=occs
#            if occs:
#                occurancesGroupedByLattice = [(getLocationFromLid(lid.replace('_', ' ')), len(sorted(zip(*occs)[1]))) for lid, occs in groupby(sorted([(getLatticeLid(l, LATTICE_ACCURACY), t) for l, t in totalOccurances], key=itemgetter(0)), key=itemgetter(0))]
#                occurancesGroupedByLattice = sorted(occurancesGroupedByLattice, key=itemgetter(1))
#                points, colors = zip(*occurancesGroupedByLattice)
#                ax = plt.subplot(111)
#                cm = matplotlib.cm.get_cmap('cool')
#                sc = plotPointsOnWorldMap(points, c=colors, cmap=cm, lw = 0)
#        #        plotPointsOnWorldMap([source], c='g', alpha=0.5, s=50, lw=1)
#        #        plt.title('%s (%s of %s) (%0.3f - %d)'%(hashtagObject['h'], len(occurances), numberOfOccurences, td/TIME_UNIT_IN_SECONDS, td))
#                divider = make_axes_locatable(ax)
#                cax = divider.append_axes("right", size="5%", pad=0.05)
#                plt.colorbar(sc, cax=cax)
#                plt.show()
                
hashtags = []
for hashtagObject in FileIO.iterateJsonFromFile('/mnt/chevron/kykamath/data/geo/hashtags/analysis/new_world/2_11/hashtags'):
    hashtags.append(hashtagObject['h'])
hashtags=sorted(hashtags)
for h in hashtags: FileIO.writeToFile(unicode(h).encode('utf-8'), 'hashtags')
    
    