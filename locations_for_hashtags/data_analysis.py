'''
Created on May 7, 2012

@author: krishnakamath
'''
from library.file_io import FileIO
from dateutil.relativedelta import relativedelta
from library.mrjobwrapper import runMRJob
from mr_data_analysis import MRDataAnalysis, PARAMS_DICT
from datetime import datetime


hdfs_input_folder = 'hdfs:///user/kykamath/geo/hashtags/%s/'


def iterateJsonFromFile(file):
    for data in FileIO.iterateJsonFromFile(file):
        if 'PARAMS_DICT' not in data: yield data

def getInputFiles(startTime, endTime, folderType='world'):
    current=startTime
    while current<=endTime:
        yield hdfs_input_folder%folderType+'%s_%s'%(current.year, current.month)
        current+=relativedelta(months=1)   

def mr_data_analysis(input_files_start_time, input_files_end_time):
    outputFile = 'tempf'
    runMRJob(MRDataAnalysis, outputFile, getInputFiles(input_files_start_time, input_files_end_time), jobconf={'mapred.reduce.tasks':300})
    FileIO.writeToFileAsJson(PARAMS_DICT, outputFile)

if __name__ == '__main__':
    input_files_start_time, input_files_end_time = datetime(2011, 2, 1), datetime(2011, 2, 27)
    mr_data_analysis(input_files_start_time, input_files_end_time)
