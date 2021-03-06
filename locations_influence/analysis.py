'''
Created on Apr 14, 2012

@author: kykamath
'''
from settings import hdfs_input_folder, location_objects_file, \
    f_ltuo_location_and_ltuo_hashtag_and_occurrence_time, \
    f_hashtag_objects
from datetime import datetime
from dateutil.relativedelta import relativedelta
from library.mrjobwrapper import runMRJob
from library.file_io import FileIO
from mr_analysis import MRAnalysis, \
    PARAMS_DICT, START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER

def iterateJsonFromFile(file):
    for data in FileIO.iterateJsonFromFile(file):
        if 'PARAMS_DICT' not in data: yield data

def getInputFiles(startTime, endTime, folderType='world'):
    current=startTime
    while current<=endTime:
        yield hdfs_input_folder%folderType+'%s_%s'%(current.year, current.month)
        current+=relativedelta(months=1)   

def mr_analysis(startTime, endTime, outputFolder, inputFilesStartTime=None, inputFilesEndTime=None):
    if not inputFilesStartTime: inputFilesStartTime=startTime; inputFilesEndTime=endTime
    outputFile = f_hashtag_objects%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
#    outputFile = location_objects_file%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
#    outputFile = f_ltuo_location_and_ltuo_hashtag_and_occurrence_time%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
    runMRJob(MRAnalysis, outputFile, getInputFiles(inputFilesStartTime, inputFilesEndTime), jobconf={'mapred.reduce.tasks':300})
    FileIO.writeToFileAsJson(PARAMS_DICT, outputFile)

if __name__ == '__main__':
    inputFilesStartTime, inputFilesEndTime = datetime(2011, 4, 1), datetime(2012, 3, 31)
    INPUT_START_TIME, INPUT_END_TIME = START_TIME, END_TIME 
    mr_analysis(INPUT_START_TIME, INPUT_END_TIME, WINDOW_OUTPUT_FOLDER, inputFilesStartTime=inputFilesStartTime, inputFilesEndTime=inputFilesEndTime)
