'''
Created on Dec 7, 2011

@author: kykamath
'''
import sys
sys.path.append('../')
from itertools import groupby
from operator import itemgetter
from library.file_io import FileIO
from experiments.mr_area_analysis import getOccuranesInHighestActiveRegion,\
    LATTICE_ACCURACY, getOccurranceDistributionInEpochs
from library.geo import getLocationFromLid, getLatticeLid, plotPointsOnWorldMap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
                        plotPointsOnWorldMap(points, blueMarble=True, bkcolor='#CFCFCF', c=colors, lw = 0)
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
    def spatialLocality():
        pass
    @staticmethod
    def run():
        pass
        
if __name__ == '__main__':
    PlotGraphsOnMap.run()
    
    