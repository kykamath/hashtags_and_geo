'''
Created on Feb 1, 2012

@author: kykamath
'''
import sys
sys.path.append('../')
from library.classes import GeneralMethods
from library.mrjobwrapper import runMRJob
from library.file_io import FileIO
from checkins.mr_modules import MRCheckins, PARAMS_DICT, BOUNDARY_ID,\
    MINIMUM_NUMBER_OF_CHECKINS_PER_USER, MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION,\
    MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION_PER_USER
from datetime import datetime
from checkins.settings import checkinsJSONFile, userToCheckinsMapFile,\
    hdfsInputCheckinsFile, FOURSQUARE_ID, GOWALLA_ID, BRIGHTKITE_ID,\
    lidsToDistributionInSocialNetworksMapFile,\
    location_objects_with_minumum_checkins_at_both_location_and_users_file,\
    checkins_graph_file


def iterateJsonFromFile(file):
    for data in FileIO.iterateJsonFromFile(file):
        if 'PARAMS_DICT' not in data: yield data

class RawDataProcessing():
    @staticmethod    
    def convert4SqDataToJSON():
        def parse4SQData(line):
            data = line.strip().split('\t')
            if len(data)!=7: data.append(None) 
            if len(data)==7: return {'id':int(data[1]), 'u': int(data[0]), 'l': [float(data[2]), float(data[3])], 't': GeneralMethods.getEpochFromDateTimeObject(datetime.strptime(data[4], '%Y-%m-%d %H:%M:%S')), 'x': data[5], 'lid': data[6]}
        for i, data in enumerate(FileIO.iterateLinesFromFile('/mnt/chevron/kykamath/data/geo/4sq/checkin_data.txt')):
            FileIO.writeToFileAsJson(parse4SQData(data), checkinsJSONFile%FOURSQUARE_ID)
    @staticmethod
    def parseJSONForGowallaAndBrightkite(line, uid=GOWALLA_ID):
        data = line.strip().split()
        return {'u': '%s_%s'%(uid, data[0]), 'l': [float(data[2]), float(data[3])], 't': GeneralMethods.getEpochFromDateTimeObject(datetime.strptime(data[1], '%Y-%m-%dT%H:%M:%SZ')), 'lid': data[4]}
    @staticmethod
    def convertBrightkiteDataToJSON():
        print 'Writing to: ', checkinsJSONFile%BRIGHTKITE_ID
        for i, line in enumerate(FileIO.iterateLinesFromFile('/mnt/chevron/kykamath/data/geo/checkins/raw_data/brightkite/loc-brightkite_totalCheckins.txt')):
            try:
                FileIO.writeToFileAsJson(RawDataProcessing.parseJSONForGowallaAndBrightkite(line, uid=BRIGHTKITE_ID), checkinsJSONFile%BRIGHTKITE_ID)
            except: pass
    @staticmethod
    def convertGowallaDataToJSON():
        print 'Writing to: ', checkinsJSONFile%GOWALLA_ID
        for i, line in enumerate(FileIO.iterateLinesFromFile('/mnt/chevron/kykamath/data/geo/checkins/raw_data/gowalla/loc-gowalla_totalCheckins.txt')):
            FileIO.writeToFileAsJson(RawDataProcessing.parseJSONForGowallaAndBrightkite(line), checkinsJSONFile%GOWALLA_ID)
    
def mr_driver(boundary_id, minimum_number_of_checkins_per_user, minimum_number_of_checkins_per_location, minimum_number_of_checkins_per_location_per_user):
    def getInputFiles(): return map(lambda id: hdfsInputCheckinsFile%id, [GOWALLA_ID, BRIGHTKITE_ID, FOURSQUARE_ID])
#    output_file = userToCheckinsMapFile%(boundary_id, minimum_number_of_checkins_per_user, minimum_number_of_checkins_per_location, minimum_number_of_checkins_per_location_per_user)
#    output_file = lidsToDistributionInSocialNetworksMapFile%(boundary_id, minimum_number_of_checkins_per_user, minimum_number_of_checkins_per_location, minimum_number_of_checkins_per_location_per_user)
    output_file = location_objects_with_minumum_checkins_at_both_location_and_users_file%(boundary_id, minimum_number_of_checkins_per_user, minimum_number_of_checkins_per_location, minimum_number_of_checkins_per_location_per_user)
#    output_file = checkins_graph_file%(boundary_id, minimum_number_of_checkins_per_user, minimum_number_of_checkins_per_location, minimum_number_of_checkins_per_location_per_user)
    runMRJob(MRCheckins, output_file, getInputFiles(), jobconf={'mapred.reduce.tasks':60})
    FileIO.writeToFileAsJson(PARAMS_DICT, output_file)

if __name__ == '__main__':
    boundary_id, minimum_number_of_checkins_per_user, minimum_number_of_checkins_per_location = BOUNDARY_ID, MINIMUM_NUMBER_OF_CHECKINS_PER_USER, MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION
    minimum_number_of_checkins_per_location_per_user = MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION_PER_USER
    mr_driver(boundary_id, minimum_number_of_checkins_per_user, minimum_number_of_checkins_per_location, minimum_number_of_checkins_per_location_per_user)
#    RawDataProcessing.convertBrightkiteDataToJSON()