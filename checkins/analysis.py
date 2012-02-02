'''
Created on Feb 1, 2012

@author: kykamath
'''
import sys
sys.path.append('../')
from library.classes import GeneralMethods
from library.mrjobwrapper import runMRJob
from library.file_io import FileIO
from checkins.mr_modules import MRCheckins
from datetime import datetime
from checkins.settings import checkinsJSONFile, userToCheckinsMapFile,\
    hdfsInputCheckinsFile, FOURSQUARE_ID, GOWALLA_ID

class RawDataProcessing():
    @staticmethod    
    def convert4SqDataToJSON():
        def parse4SQData(line):
            data = line.strip().split('\t')
            if len(data)!=7: data.append(None) 
            if len(data)==7: return {'id':int(data[1]), 'u': int(data[0]), 'l': [float(data[2]), float(data[3])], 't': GeneralMethods.getEpochFromDateTimeObject(datetime.strptime(data[4], '%Y-%m-%d %H:%M:%S')), 'x': data[5], 'lid': data[6]}
        for i, data in enumerate(FileIO.iterateLinesFromFile('/mnt/chevron/kykamath/data/geo/4sq/checkin_data.txt')):
            print i
            FileIO.writeToFileAsJson(parse4SQData(data), checkinsJSONFile%FOURSQUARE_ID)
    @staticmethod
    def parseJSONForGowallaAndBrightkite(line):
        print line
        data = line.strip().split()
        print {'u': '%s_%s'%(GOWALLA_ID, data[0]), 'l': [float(data[2]), float(data[3])], 't': GeneralMethods.getEpochFromDateTimeObject(datetime.strptime(data[4], '%Y-%m-%dT%H:%M:%SZ')), 'lid': data[4]}

    @staticmethod
    def convertGowallaDataToJSON():
        for i, line in enumerate(FileIO.iterateLinesFromFile('/mnt/chevron/kykamath/data/geo/checkins/raw_data/brightkite/loc-brightkite_totalCheckins.txt')):
            RawDataProcessing.parseJSONForGowallaAndBrightkite(line)
            if i==10: exit()
    
def mr_driver():
    runMRJob(MRCheckins, userToCheckinsMapFile, [hdfsInputCheckinsFile], jobconf={'mapred.reduce.tasks':60})

if __name__ == '__main__':
#    mr_driver()
    RawDataProcessing.convertGowallaDataToJSON()