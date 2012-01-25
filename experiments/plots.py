'''
Created on Dec 7, 2011

@author: kykamath
'''
import sys
from settings import hashtagsLatticeGraphFile, hashtagsFile
from experiments.models import filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance
from library.stats import getOutliersRangeUsingIRQ
from library.plotting import getLatexForString
sys.path.append('../')
from itertools import groupby
from operator import itemgetter
from library.file_io import FileIO
from experiments.mr_area_analysis import getOccuranesInHighestActiveRegion,\
    LATTICE_ACCURACY, getOccurranceDistributionInEpochs, getRadius
from library.geo import getLocationFromLid, getLatticeLid, plotPointsOnWorldMap,\
    getHaversineDistanceForLids
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
import numpy as np
from datetime import datetime
import scipy.stats

RIO_DE_JANEIRO = '-22.7650_-43.0650'
NEW_YORK = '40.6000_-73.8050'
SAO_PAULO = '-23.4900_-46.5450'

class PlotGraphsOnMap:
    @staticmethod
    def plotGraphsForHashtag(hashtag):
        for hashtagObject in FileIO.iterateJsonFromFile('/mnt/chevron/kykamath/data/geo/hashtags/analysis/all_world/2_11/hashtagsWithoutEndingWindow'):
            MINUTES = 5
            if hashtagObject['h']==hashtag:
                print unicode(hashtagObject['h']).encode('utf-8'), len(hashtagObject['oc'])
                occsDistributionInTimeUnits = getOccurranceDistributionInEpochs(getOccuranesInHighestActiveRegion(hashtagObject), timeUnit=MINUTES*60, fillInGaps=True, occurancesCount=False)
                totalOccurances = []
                for interval, t in enumerate(sorted(occsDistributionInTimeUnits)):
                    occs = occsDistributionInTimeUnits[t]
                    totalOccurances+=occs
                    if occs:
                        fileName = '../images/plotsOnMap/%s/%s.png'%(hashtagObject['h'], (interval+1)*MINUTES); FileIO.createDirectoryForFile(fileName)
                        print fileName
                        occurancesGroupedByLattice = [(getLocationFromLid(lid.replace('_', ' ')), 'm') for lid, occs in groupby(sorted([(getLatticeLid(l, LATTICE_ACCURACY), t) for l, t in totalOccurances], key=itemgetter(0)), key=itemgetter(0))]
                        occurancesGroupedByLattice = sorted(occurancesGroupedByLattice, key=itemgetter(1))
                        points, colors = zip(*occurancesGroupedByLattice)
                        plotPointsOnWorldMap(points, blueMarble=False, bkcolor='#CFCFCF', c=colors, lw = 0)
                        plt.show()
    #                    plt.savefig(fileName)
                        plt.clf()
                    if (interval+1)*MINUTES>=120: break
                break
    @staticmethod
    def writeHashtagsFile():
        hashtags = []
        for hashtagObject in FileIO.iterateJsonFromFile('/mnt/chevron/kykamath/data/geo/hashtags/analysis/all_world/2_11/hashtagsWithoutEndingWindow'):
            print hashtagObject.keys()
            exit()
            hashtags.append(hashtagObject['h'])
        hashtags=sorted(hashtags)
        for h in hashtags: FileIO.writeToFile(unicode(h).encode('utf-8'), 'hashtags')
    @staticmethod
    def run():
        PlotGraphsOnMap.plotGraphsForHashtag('chupacorinthians')
        PlotGraphsOnMap.plotGraphsForHashtag('cnndebate')    
        PlotGraphsOnMap.plotGraphsForHashtag('ripstevejobs')
        PlotGraphsOnMap.plotGraphsForHashtag('ripamywinehouse')
    
class Locality:
    ''' Total memes (Mar-Oct): 3057
    Train memes (Mar-Aug): 1466
    Test memes (Sept-Oct): 515
    '''
    @staticmethod
    def _getDistances():
        distances = {}
        for latticeObject in FileIO.iterateJsonFromFile(hashtagsLatticeGraphFile%('training_world','%s_%s'%(2,11))):
            latticeHashtagsSet = set(latticeObject['hashtags'])
            for neighborLattice, neighborHashtags in latticeObject['links'].iteritems():
                key = '_'.join(sorted([latticeObject['id'], neighborLattice]))
                if key not in distances:
                    distances[key] = {}
                    neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags, findLag=False)
                    neighborHashtagsSet = set(neighborHashtags)
                    distances[key]['similarity']=len(latticeHashtagsSet.intersection(neighborHashtagsSet))/float(len(latticeHashtagsSet.union(neighborHashtagsSet)))
                    distances[key]['temporalDistance']=np.mean([abs(latticeObject['hashtags'][k][0]-neighborHashtags[k][0]) for k in neighborHashtags if k in latticeObject['hashtags']])
                    distances[key]['geoDistance']=getHaversineDistanceForLids(latticeObject['id'].replace('_', ' '), neighborLattice.replace('_', ' '))
        return distances
#    @staticmethod
#    def plotTemporalLocality():
#        distances = Locality._getDistances()
#        dataToPlot = defaultdict(list)
#        for _, data in distances.iteritems():
#            dataToPlot[round(data['similarity'],2)].append(data['temporalDistance']) 
#        for k in sorted(dataToPlot):
#            _, upperRange = getOutliersRangeUsingIRQ(dataToPlot[k])
#            print k, len(dataToPlot[k]), len(filter(lambda i:i<upperRange, dataToPlot[k]))
#            plt.scatter(k, np.mean(filter(lambda i:i<upperRange, dataToPlot[k]))/(60.*60.), c='r', lw = 0)
##        plt.show()
#        plt.title('Temporal distance between lattices'), plt.xlabel('Jaccard similarity'), plt.ylabel('Mean time difference (hours)')
#        plt.savefig('../images/temporalDistance.png')
#    @staticmethod
#    def plotSpatialLocality():
#        distances = Locality._getDistances()
#        dataToPlot = defaultdict(list)
#        for _, data in distances.iteritems():
#            dataToPlot[round(data['similarity'],2)].append(data['geoDistance']) 
#        for k in sorted(dataToPlot):
#            _, upperRange = getOutliersRangeUsingIRQ(dataToPlot[k])
#            print k, len(dataToPlot[k]), len(filter(lambda i:i<upperRange, dataToPlot[k]))
#            plt.scatter(k, np.mean(filter(lambda i:i<upperRange, dataToPlot[k])), c='r', lw = 0)
##        plt.show()
#        plt.title('Haversine distance between lattices'), plt.xlabel('Jaccard similarity'), plt.ylabel('Mean haversine distance (miles)')
#        plt.savefig('../images/geoDistance.png')
    @staticmethod
    def temporalLocalitySimilarityExample(lattice=NEW_YORK):
        distances = defaultdict(dict)
        for latticeObject in FileIO.iterateJsonFromFile(hashtagsLatticeGraphFile%('training_world','%s_%s'%(2,11))):
            if latticeObject['id']==lattice:
                latticeHashtagsSet = set(latticeObject['hashtags'])
                for neighborLattice, neighborHashtags in latticeObject['links'].iteritems():
                    distances[neighborLattice] = {}
                    neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags, findLag=False)
                    neighborHashtagsSet = set(neighborHashtags)
                    distances[neighborLattice]['similarity']=len(latticeHashtagsSet.intersection(neighborHashtagsSet))/float(len(latticeHashtagsSet.union(neighborHashtagsSet)))
                    distances[neighborLattice]['temporalDistance']=np.mean([abs(latticeObject['hashtags'][k][0]-neighborHashtags[k][0]) for k in neighborHashtags if k in latticeObject['hashtags']])/(60.*60.)
                    distances[neighborLattice]['geoDistance']=getHaversineDistanceForLids(latticeObject['id'].replace('_', ' '), neighborLattice.replace('_', ' '))
                break
        dataPoints = []
        ax = plt.subplot(111)
        for k, data in distances.iteritems(): dataPoints.append((getLocationFromLid(k.replace('_', ' ')), data['similarity']))
        points, colors = zip(*sorted(dataPoints, key=itemgetter(1)))
        sc = plotPointsOnWorldMap(points, blueMarble=False, bkcolor='#CFCFCF', cmap='RdPu', c=colors, lw = 0, alpha=1.0)
        plotPointsOnWorldMap([getLocationFromLid(lattice.replace('_', ' '))], blueMarble=False, bkcolor='#CFCFCF', c='#64FF1C', lw = 0)
        divider = make_axes_locatable(ax)
        plt.title('Jaccard similarity with New York')
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc, cax=cax)
#        plt.show()
        plt.savefig('../images/similarityExample.png')
    @staticmethod
    def temporalLocalityTemporalDistanceExample(lattice=NEW_YORK):
        distances = defaultdict(dict)
        for latticeObject in FileIO.iterateJsonFromFile(hashtagsLatticeGraphFile%('training_world','%s_%s'%(2,11))):
            if latticeObject['id']==lattice:
                latticeHashtagsSet = set(latticeObject['hashtags'])
                for neighborLattice, neighborHashtags in latticeObject['links'].iteritems():
                    distances[neighborLattice] = {}
                    neighborHashtags = filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance(latticeObject['hashtags'], neighborHashtags, findLag=False)
                    neighborHashtagsSet = set(neighborHashtags)
                    distances[neighborLattice]['similarity']=len(latticeHashtagsSet.intersection(neighborHashtagsSet))/float(len(latticeHashtagsSet.union(neighborHashtagsSet)))
                    distances[neighborLattice]['temporalDistance']=np.mean([abs(latticeObject['hashtags'][k][0]-neighborHashtags[k][0]) for k in neighborHashtags if k in latticeObject['hashtags']])/(60.*60.)
                    distances[neighborLattice]['geoDistance']=getHaversineDistanceForLids(latticeObject['id'].replace('_', ' '), neighborLattice.replace('_', ' '))
                break
        dataPoints = []
        ax = plt.subplot(111)
        for k, data in distances.iteritems(): dataPoints.append((getLocationFromLid(k.replace('_', ' ')), data['temporalDistance']))
        points, colors = zip(*sorted(dataPoints, key=itemgetter(1)))
        sc = plotPointsOnWorldMap(points, blueMarble=False, bkcolor='#CFCFCF', cmap='RdPu', c=colors, lw = 0, alpha=1.0)
        plotPointsOnWorldMap([getLocationFromLid(lattice.replace('_', ' '))], blueMarble=False, bkcolor='#CFCFCF', c='#64FF1C', lw = 0)
        divider = make_axes_locatable(ax)
        plt.title('Average time difference from New York')
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc, cax=cax)
#        plt.show()
        plt.savefig('../images/temporalDistanceExample.png')
    @staticmethod
    def plotTemporalLocality():
        distances = Locality._getDistances()
        dataToPlot, dataX, dataY = defaultdict(list), [], []
        ax = plt.gca()
        for _, data in distances.iteritems():
            dataToPlot[int(data['geoDistance'])/100*100+100].append(data['temporalDistance']/(60*60)) 
        
        for k in sorted(dataToPlot):
            _, upperRange = getOutliersRangeUsingIRQ(dataToPlot[k])
            points = filter(lambda i:i<upperRange, dataToPlot[k])
            if len(points)>50:
                dataX.append(k), dataY.append(np.mean(points))
        pearsonCoeff, p_value = scipy.stats.pearsonr(dataX, dataY)
#        print round(pearsonCoeff,2), round(p_value, 2)
        plt.scatter(dataX, dataY, c='r', lw = 0)
        plt.title('Temporal distance between lattices ' + getLatexForString('( \\rho = %0.2f, p-value = %0.2f )'%(pearsonCoeff, p_value))), plt.xlabel('Haversine distance (miles)'), plt.ylabel('Temporal distance (hours)')
#        plt.show()
        plt.savefig('../images/temporalLocality.png')
    @staticmethod
    def plotSpatialLocality():
        distances = Locality._getDistances()
        dataToPlot, dataX, dataY = defaultdict(list), [], []
        ax = plt.gca()
        for _, data in distances.iteritems():
            dataToPlot[int(data['geoDistance'])/100*100+100].append(data['similarity']) 
        
        for k in sorted(dataToPlot):
            _, upperRange = getOutliersRangeUsingIRQ(dataToPlot[k])
            points = filter(lambda i:i<upperRange, dataToPlot[k])
            if len(points)>50:
#                print k, len(dataToPlot[k]), len(points)
                dataX.append(k), dataY.append(np.mean(points))
        pearsonCoeff, p_value = scipy.stats.pearsonr(dataX, dataY)
        print round(pearsonCoeff,2), round(p_value, 2)
        plt.scatter(dataX, dataY, c='r', lw = 0)
        plt.title('Similarity between lattices ' + getLatexForString('( \\rho = %0.2f, p-value = %0.2f )'%(pearsonCoeff, p_value))), plt.xlabel('Haversine distance (miles)'), plt.ylabel('Jaccard similarity')
#        plt.show()
        plt.savefig('../images/spatialLocality.png')    
    
    @staticmethod
    def run():
        Locality.plotTemporalLocality()
#        Locality.plotSpatialLocality()
#        Locality.temporalLocalitySimilarityExample()
#        Locality.temporalLocalityTemporalDistanceExample()
        
class Coverage:
    @staticmethod
    def temp():
        MINUTES = 5
        for timeUnit in [1, 3, 6]:
            print timeUnit
            data = defaultdict(int)
            for hashtagObject in FileIO.iterateJsonFromFile(hashtagsFile%('training_world','%s_%s'%(2,11))):
                try:
                    occsDistributionInTimeUnits = getOccurranceDistributionInEpochs(getOccuranesInHighestActiveRegion(hashtagObject), timeUnit=MINUTES*60, fillInGaps=True, occurancesCount=False)
                    occurances = list(zip(*sorted(occsDistributionInTimeUnits.iteritems(), key=itemgetter(0)))[1])
                    occsInTimeunit =  zip(*reduce(lambda aggList, l: aggList+l, occurances[:timeUnit], []))[0]
                    if len(occsInTimeunit)>10:
                        allOccurances = zip(*reduce(lambda aggList, l: aggList+l, occurances, []))[0]
                        timeUnitRadius, allRadius = getRadius(occsInTimeunit), getRadius(allOccurances)
                        data[int(abs(timeUnitRadius-allRadius))/50*50+50]+=1
                except IndexError as e: pass
            dataX, dataY = zip(*sorted(data.iteritems(), key=itemgetter(0)))
            plt.loglog(dataX, dataY, lw=2, label=str(timeUnit*MINUTES))
        plt.legend()
        plt.show()
            
    @staticmethod
    def run():
        Coverage.temp()

if __name__ == '__main__':
#    PlotGraphsOnMap.run()
#    Locality.run()
    Coverage.run()
#    print getLatticeLid([-23.549569,-46.639173],  0.145)
    