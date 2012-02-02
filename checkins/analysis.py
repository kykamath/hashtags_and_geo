'''
Created on Feb 1, 2012

@author: kykamath
'''
import sys
from library.classes import GeneralMethods
sys.path.append('../')
from library.file_io import FileIO
from datetime import datetime
from checkins.settings import checkinsJSONFile
def parseData(line):
    data = line.strip().split('\t')
    if len(data)!=7: data.append(None) 
    if len(data)==7: return {'id':int(data[1]), 'u': int(data[0]), 'l': [float(data[2]), float(data[3])], 't': GeneralMethods.getEpochFromDateTimeObject(datetime.strptime(data[4], '%Y-%m-%d %H:%M:%S')), 'x': data[5], 'lid': data[6]}

def writeCheckinsToJSONFormat():
    for i, data in enumerate(FileIO.iterateLinesFromFile('/mnt/chevron/kykamath/data/geo/checkin_data.txt')):
#    for i, data in enumerate(FileIO.iterateLinesFromFile('../data/checkin_data.txt')):
        print i
        FileIO.writeToFileAsJson(parseData(data), checkinsJSONFile)
        
writeCheckinsToJSONFormat()