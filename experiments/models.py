'''
Created on Dec 8, 2011

@author: kykamath
'''
from library.file_io import FileIO
from settings import hashtagsWithoutEndingWindowFile
from experiments.mr_area_analysis import getOccuranesInHighestActiveRegion

class Customer:
    pass

class Graph:
    pass

class Hashtag:
    def __init__(self, hashtagObject): self.hashtagObject = hashtagObject
    def getNextOccurance(self):
        for oc in getOccuranesInHighestActiveRegion(self.hashtagObject): yield oc
    @staticmethod
    def iterateHashtags(timeRange, folderType):
        for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(folderType,'%s_%s'%timeRange)): yield Hashtag(h)

if __name__ == '__main__':
    folderType = 'world'
    timeRange = (2,11)
    i=1
    for h in Hashtag.iterateHashtags(timeRange, folderType):
        for oc in h.getNextOccurance():
            print i, oc; i+=1
#        exit()