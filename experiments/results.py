'''
Created on Nov 24, 2011

@author: kykamath
'''
from library.file_io import FileIO
from settings import hashtagsAnalayzeLocalityIndexAtKFile
import matplotlib.pyplot as plt
from operator import itemgetter

class AnalyzeLocalityIndexAtK:
    @staticmethod
    def LIForOccupy(timeRange):
        imagesFolder = '/tmp/images/'
        occupyList = ['occupywallst', 'occupyoakland', 'occupydc', 'occupysf']
        FileIO.createDirectoryForFile(imagesFolder+'dsf')
        for h in FileIO.iterateJsonFromFile(hashtagsAnalayzeLocalityIndexAtKFile%'%s_%s'%timeRange):
    #        if h['h'].startswith('occupy') or h['h']=='ows':
            if h['h'] in occupyList:
                print h['h']
                dataX, dataY = zip(*[(k, v[0])for k, v in h['liAtVaryingK']])
                plt.plot(dataX, dataY, label=h['h'])
    #    plt.title(h['h'])
        plt.ylim(ymax=4000)
        plt.legend(loc=2)
        plt.show()
    @staticmethod
    def rankHashtagsBYLIScore(timeRange):
        hashtagsLIScore = [(h['h'], h['liAtVaryingK'][0][1][0]) 
        for h in FileIO.iterateJsonFromFile(hashtagsAnalayzeLocalityIndexAtKFile%'%s_%s'%timeRange)]
        for h, s in sorted(hashtagsLIScore, key=itemgetter(1)):
            print h, s
if __name__ == '__main__':
    timeRange = (2,11)
#    AnalyzeLocalityIndexAtK.LIForOccupy(timeRange)
    AnalyzeLocalityIndexAtK.rankHashtagsBYLIScore(timeRange)