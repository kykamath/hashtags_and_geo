'''
Created on Dec 7, 2011

@author: kykamath
'''
import sys
from library.classes import GeneralMethods
import matplotlib
sys.path.append('../')
from settings import hashtagsLatticeGraphFile, hashtagsFile
from models import filterOutNeighborHashtagsOutside1_5IQROfTemporalDistance,\
    getLattices, CoverageBasedLatticeSelectionModel
from library.stats import getOutliersRangeUsingIRQ
from library.plotting import getLatexForString
from itertools import groupby
from operator import itemgetter
from library.file_io import FileIO
from mr_area_analysis import getOccuranesInHighestActiveRegion,\
    LATTICE_ACCURACY, getOccurranceDistributionInEpochs, getRadius
from library.geo import getLocationFromLid, getLatticeLid, plotPointsOnWorldMap,\
    getHaversineDistanceForLids, getHaversineDistance
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
import numpy as np
from datetime import datetime
import scipy.stats
from scipy.stats import ks_2samp

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
    def temp():
        hashtags, MINUTES = [], 60
        for hashtagObject in FileIO.iterateJsonFromFile('americanhorrorstory'):
            if hashtagObject['h']=='americanhorrorstory':
                print unicode(hashtagObject['h']).encode('utf-8'), len(hashtagObject['oc'])
                occsDistributionInTimeUnits = getOccurranceDistributionInEpochs(getOccuranesInHighestActiveRegion(hashtagObject, timeUnit=60*60), timeUnit=MINUTES*60, fillInGaps=True, occurancesCount=False)
                totalOccurances = []
                for interval, t in enumerate(sorted(occsDistributionInTimeUnits)):
                    occs = occsDistributionInTimeUnits[t]
                    if occs:
                        fileName = '../images/plotsOnMap/%s/%s.png'%(hashtagObject['h'], (interval+1)*MINUTES); FileIO.createDirectoryForFile(fileName)
#                        print interval, t, len(occs)
                        print fileName
                        occurancesGroupedByLattice = [(getLocationFromLid(lid.replace('_', ' ')), 'm') for lid, occ in groupby(sorted([(getLatticeLid(l, LATTICE_ACCURACY), t) for l, t in occs], key=itemgetter(0)), key=itemgetter(0))]
                        occurancesGroupedByLattice = sorted(occurancesGroupedByLattice, key=itemgetter(1))
                        points, colors = zip(*occurancesGroupedByLattice)
                        plotPointsOnWorldMap(points, blueMarble=False, bkcolor='#CFCFCF', c=colors, lw = 0)
#                        plt.show()
                        plt.savefig(fileName)
                        plt.clf()
                exit()
    @staticmethod
    def run():
#        PlotGraphsOnMap.plotGraphsForHashtag('chupacorinthians')
#        PlotGraphsOnMap.plotGraphsForHashtag('cnndebate')    
#        PlotGraphsOnMap.plotGraphsForHashtag('ripstevejobs')
#        PlotGraphsOnMap.plotGraphsForHashtag('ripamywinehouse')
        PlotGraphsOnMap.temp()
    
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
        print round(pearsonCoeff,2), round(p_value, 2)
        plt.scatter(dataX, dataY, c='r', lw = 0)
#        plt.title('Temporal distance between lattices ' + getLatexForString('( \\rho = %0.2f, p-value = %0.2f )'%(pearsonCoeff, p_value)))
        plt.xlabel('Spatial affinity (miles)', fontsize=20), plt.ylabel('Temporal affinity (hours)', fontsize=20)
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
#        plt.title('Similarity between lattices ' + getLatexForString('( \\rho = %0.2f, p-value = %0.2f )'%(pearsonCoeff, p_value))), 
        plt.xlabel('Spatial affinity (miles)', fontsize=20), plt.ylabel('Content affinity', fontsize=20)
#        plt.show()
        plt.savefig('../images/spatialLocality.png')    
    
    @staticmethod
    def run():
#        Locality.plotTemporalLocality()
        Locality.plotSpatialLocality()
#        Locality.temporalLocalitySimilarityExample()
#        Locality.temporalLocalityTemporalDistanceExample()
        
class Coverage:
    @staticmethod
    def coverageIndication():
        MINUTES = 5
        for timeUnit, color, shape in [(1, 'r', 'x'), (3, 'g', 'd'), (6, 'b', 's')]:
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
#                        data[round(abs(timeUnitRadius-allRadius)/allRadius, 2)]+=1
                except IndexError as e: pass
            for k in data.keys()[:]: 
                if data[k]<3: del data[k]
            dataX, dataY = zip(*sorted(data.iteritems(), key=itemgetter(0)))
            plt.loglog(dataX, dataY, lw=2, label=str(timeUnit*MINUTES) + ' minutes', marker=shape)
#        plt.loglog([1],[1])
#        plt.title('Early indication of coverage'), 
        plt.xlabel('Coverage difference (miles)', fontsize=20), plt.ylabel('Number of hashtags', fontsize=20)
        plt.legend()
#        plt.show()
        plt.savefig('../images/coverageIndication.png')
    @staticmethod
    def probabilisticCoverageModelExample(hashtag, type):
        MINUTES, timeUnit = 5, 1
        print len(CoverageBasedLatticeSelectionModel.lattices)
        for hashtagObject in FileIO.iterateJsonFromFile('/mnt/chevron/kykamath/data/geo/hashtags/analysis/all_world/2_11/hashtagsWithoutEndingWindow'):
            if hashtagObject['h']==hashtag:
                occsDistributionInTimeUnits = getOccurranceDistributionInEpochs(getOccuranesInHighestActiveRegion(hashtagObject), timeUnit=MINUTES*60, fillInGaps=True, occurancesCount=False)
                occurances = list(zip(*sorted(occsDistributionInTimeUnits.iteritems(), key=itemgetter(0)))[1])
                occsInTimeunit =  zip(*reduce(lambda aggList, l: aggList+l, occurances[:timeUnit], []))[0]
                allOccurances = zip(*reduce(lambda aggList, l: aggList+l, occurances, []))[0]
                if type=='5m': probabilityDistributionForObservedLattices = CoverageBasedLatticeSelectionModel.probabilityDistributionForLattices(occsInTimeunit)
                else: 
                    print getRadius(allOccurances)
                    probabilityDistributionForObservedLattices = CoverageBasedLatticeSelectionModel.probabilityDistributionForLattices(allOccurances)
                latticeScores = CoverageBasedLatticeSelectionModel.spreadProbability(CoverageBasedLatticeSelectionModel.lattices, probabilityDistributionForObservedLattices)
                points, colors = zip(*map(lambda t: (getLocationFromLid(t[0].replace('_', ' ')), t[1]), sorted(latticeScores.iteritems(), key=itemgetter(1))))
#                print points[0], colors[0]
                ax = plt.subplot(111)
                sc = plotPointsOnWorldMap(points, blueMarble=False, bkcolor='#CFCFCF', c=colors, cmap='cool', lw = 0)
                divider = make_axes_locatable(ax)
#                plt.title('Jaccard similarity with New York')
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(sc, cax=cax)
                plt.show()
#                plt.savefig('../images/coverage_examples/%s_%s.png'%(hashtag, type))
                plt.clf()
                break
    @staticmethod
    def temp(hashtag='blackparentsquotes'):
        for hashtagObject in FileIO.iterateJsonFromFile('/mnt/chevron/kykamath/data/geo/hashtags/analysis/all_world/2_11/hashtagsWithoutEndingWindow'):
#            print hashtagObject['h']
            if hashtagObject['h']==hashtag:
                print hashtagObject['h']
                occsDistributionInTimeUnits = getOccurranceDistributionInEpochs(hashtagObject['oc'], timeUnit=5, fillInGaps=True, occurancesCount=False)
#                plt.plot_date()
                exit()
                
#        MINUTES = 5
##        for timeUnit, color, shape in [(1, 'r', 'x'), (3, 'g', 'd'), (6, 'b', 's')]:
#        for timeUnit, color, shape in [(6, 'r', 'x')]:
#            print timeUnit
#            data = defaultdict(int)
#            for hashtagObject in FileIO.iterateJsonFromFile(hashtagsFile%('training_world','%s_%s'%(2,11))):
#                try:
#                    occsDistributionInTimeUnits = getOccurranceDistributionInEpochs(getOccuranesInHighestActiveRegion(hashtagObject), timeUnit=MINUTES*60, fillInGaps=True, occurancesCount=False)
#                    occurances = list(zip(*sorted(occsDistributionInTimeUnits.iteritems(), key=itemgetter(0)))[1])
#                    occsInTimeunit =  zip(*reduce(lambda aggList, l: aggList+l, occurances[:timeUnit], []))[0]
#                    if len(occsInTimeunit)>10:
#                        allOccurances = zip(*reduce(lambda aggList, l: aggList+l, occurances, []))[0]
##                        timeUnitRadius, allRadius = getRadius(occsInTimeunit), getRadius(allOccurances)
#                        timeUnitDist = CoverageBasedLatticeSelectionModel.probabilityDistributionForLattices(occsInTimeunit)
#                        timeUnitDist = CoverageBasedLatticeSelectionModel.spreadProbability(CoverageBasedLatticeSelectionModel.lattices, timeUnitDist)
#                        allDist = CoverageBasedLatticeSelectionModel.probabilityDistributionForLattices(allOccurances)
#                        allDist = CoverageBasedLatticeSelectionModel.spreadProbability(CoverageBasedLatticeSelectionModel.lattices, allDist)
##                        print timeUnitRadius, allRadius
#                        ksstat, p_value = ks_2samp(timeUnitDist.values(),allDist.values())
#                        print hashtagObject['h'], round(ksstat,3), round(p_value,3)
##                        exit()
#                except IndexError as e: pass
            
    @staticmethod
    def run():
#        Coverage.probabilisticCoverageModelExample('ripstevejobs', '120m')
#        Coverage.probabilisticCoverageModelExample('ripstevejobs', '5m')
#        Coverage.probabilisticCoverageModelExample('cnndebate', '120m')
#        Coverage.probabilisticCoverageModelExample('cnndebate', '5m')
        Coverage.coverageIndication()
#        Coverage.temp()
        
def plot_locations_on_world_map():
    MINUTES = 15
    hashtags = ['ripstevejobs', 'cnbcdebate']
    MAP_FROM_HASHTAG_TO_SUBPLOT = dict([('ripstevejobs', 211), ('cnbcdebate', 212)])
    map_from_epoch_lag_to_map_from_hashtag_to_tuples_of_location_and_epoch_lag = defaultdict(dict)
    for hashtag in hashtags:
        for hashtag_object in FileIO.iterateJsonFromFile('./data/%s.json'%hashtag):
            map_from_epoch_time_unit_to_tuples_of_location_and_epoch_occurrence_time =  getOccurranceDistributionInEpochs(getOccuranesInHighestActiveRegion(hashtag_object), timeUnit=MINUTES*60, fillInGaps=True, occurancesCount=False)
            tuples_of_epoch_time_unit_and_tuples_of_location_and_epoch_occurrence_time = sorted(map_from_epoch_time_unit_to_tuples_of_location_and_epoch_occurrence_time.iteritems(), key=itemgetter(0))
#            GeneralMethods.runCommand('rm -rf ./images/plot_locations_on_world_map/%s'%hashtag)
    #        time_starting_time_unit = datetime.fromtimestamp(tuples_of_epoch_time_unit_and_tuples_of_location_and_epoch_occurrence_time[0][0])
            epoch_starting_time_unit = tuples_of_epoch_time_unit_and_tuples_of_location_and_epoch_occurrence_time[0][0]
            epoch_ending_time_unit = epoch_starting_time_unit+25*60*60
            for epoch_time_unit, tuples_of_location_and_epoch_occurrence_time in tuples_of_epoch_time_unit_and_tuples_of_location_and_epoch_occurrence_time:
                if epoch_time_unit<=epoch_ending_time_unit:
                    if tuples_of_location_and_epoch_occurrence_time:
        #                file_world_map_plot = './images/plot_locations_on_world_map/%s/%s.png'%(hashtag, str(datetime.fromtimestamp(epoch_time_unit) - epoch_starting_time_unit))
                        epoch_lag = epoch_time_unit - epoch_starting_time_unit
        #                print file_world_map_plot
#                        locations = zip(*tuples_of_location_and_epoch_occurrence_time)[0]
#                        tuples_of_locations_and_no_of_occurrences = [(location, len(list(iterator_of_locations)))
#                               for location, iterator_of_locations in 
#                               groupby(sorted(locations, key=itemgetter(0,1)), key=itemgetter(0,1))
#                            ]
        #                locations, colors = zip(*sorted(tuples_of_locations_and_no_of_occurrences, key=itemgetter(1)))
                        tuples_of_location_and_epoch_occurrence_time = sorted(tuples_of_location_and_epoch_occurrence_time, key=itemgetter(1))
                        map_from_epoch_lag_to_map_from_hashtag_to_tuples_of_location_and_epoch_lag[epoch_lag][hashtag] = [(getLatticeLid(location, 0.145), epoch_occurrence_time-epoch_starting_time_unit)for location, epoch_occurrence_time in tuples_of_location_and_epoch_occurrence_time]
        #                plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=colors, cmap=matplotlib.cm.cool, lw = 0)
        #                plt.show()
        #                FileIO.createDirectoryForFile(file_world_map_plot)
        #                plt.savefig(file_world_map_plot)
        #                plt.clf()
    
    map_from_hashtag_to_accumulated_tuples_of_location_and_epoch_lag = defaultdict(list)
    GeneralMethods.runCommand('rm -rf ./images/plot_locations_on_world_map/')
    for epoch_lag in sorted(map_from_epoch_lag_to_map_from_hashtag_to_tuples_of_location_and_epoch_lag):
        file_world_map_plot = './images/plot_locations_on_world_map/%s.png'%(epoch_lag)
        print file_world_map_plot
        map_from_hashtag_to_tuples_of_location_and_epoch_lag = map_from_epoch_lag_to_map_from_hashtag_to_tuples_of_location_and_epoch_lag[epoch_lag]
#        print epoch_lag, map_from_hashtag_to_tuples_of_location_and_epoch_lag.keys()
        for hashtag, tuples_of_location_and_epoch_lag in map_from_hashtag_to_tuples_of_location_and_epoch_lag.iteritems():
            map_from_hashtag_to_accumulated_tuples_of_location_and_epoch_lag[hashtag]+=tuples_of_location_and_epoch_lag
#        for k in map_from_hashtag_to_accumulated_tuples_of_location_and_epoch_occurrence_time:
#            print k, len(map_from_hashtag_to_accumulated_tuples_of_location_and_epoch_occurrence_time[k]),
        for hashtag, accumulated_tuples_of_location_and_epoch_lag in map_from_hashtag_to_accumulated_tuples_of_location_and_epoch_lag.iteritems():
#            print accumulated_tuples_of_location_and_epoch_lag
            plt.subplot(MAP_FROM_HASHTAG_TO_SUBPLOT[hashtag])
            tuples_of_location_and_epoch_max_lag= [(location, max(zip(*iterator_of_tuples_of_location_and_epoch_lag)[1]))
                               for location, iterator_of_tuples_of_location_and_epoch_lag in 
                               groupby(sorted(accumulated_tuples_of_location_and_epoch_lag, key=itemgetter(0)), key=itemgetter(0))
                            ]
            locations, colors = zip(*[(getLocationFromLid(location.replace('_', ' ')), epoch_max_lag) for location, epoch_max_lag in sorted(tuples_of_location_and_epoch_max_lag, key=itemgetter(1))])
            plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=colors, cmap=matplotlib.cm.cool, lw = 0)
            plt.title('%s      (%s hours)'%(hashtag, epoch_lag/(60.*60)))
#        plt.show()
        FileIO.createDirectoryForFile(file_world_map_plot)
        plt.savefig(file_world_map_plot)
        plt.clf()
#        exit()
    pass
        
if __name__ == '__main__':
#    PlotGraphsOnMap.run()
#    Locality.run()
#    Coverage.run()
#    print getLatticeLid([-23.549569,-46.639173],  0.145)

    plot_locations_on_world_map()
    