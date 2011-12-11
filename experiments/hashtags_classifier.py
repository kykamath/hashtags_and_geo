'''
Created on Dec 11, 2011

@author: kykamath
'''
from library.file_io import FileIO
from settings import hashtagsWithoutEndingWindowFile
timeRange, folderType = (2,11), 'world'
for h in FileIO.iterateJsonFromFile(hashtagsWithoutEndingWindowFile%(folderType,'%s_%s'%timeRange)):
    print h.keys() 