'''
Created on May 7, 2012

@author: krishnakamath
'''
from library.file_io import FileIO
from dateutil.relativedelta import relativedelta
from library.mrjobwrapper import runMRJob
from mr_data_analysis import MRDataAnalysis, PARAMS_DICT
from datetime import datetime


############################# Settings ##########################
hdfs_input_folder = 'hdfs:///user/kykamath/geo/hashtags/%s/'

fld_data_analysis = '/mnt/chevron/kykamath/data/geo/hashtags/data_analysis/%s_%s/'

f_tuo_hashtag_and_occurrences_count = fld_data_analysis+'/tuo_hashtag_and_occurrences_count'
#################################################################



def iterateJsonFromFile(file):
    for data in FileIO.iterateJsonFromFile(file):
        if 'PARAMS_DICT' not in data: yield data

def getInputFiles(startTime, endTime, folderType='world'):
    current=startTime
    while current<=endTime:
        input_file = hdfs_input_folder%folderType+'%s_%s'%(current.year, current.month)
        print input_file
        yield input_file
        current+=relativedelta(months=1)   

def mr_data_analysis(input_files_start_time, input_files_end_time):
    outputFile = f_tuo_hashtag_and_occurrences_count%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'))
    runMRJob(MRDataAnalysis, outputFile, getInputFiles(input_files_start_time, input_files_end_time), jobconf={'mapred.reduce.tasks':300})
    FileIO.writeToFileAsJson(PARAMS_DICT, outputFile)

if __name__ == '__main__':
    input_files_start_time, input_files_end_time = datetime(2011, 2, 1), datetime(2011, 2, 27)
    mr_data_analysis(input_files_start_time, input_files_end_time)
