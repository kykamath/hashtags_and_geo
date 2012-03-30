'''
Created on Nov 19, 2011

@author: kykamath
'''
from settings import hdfsInputFolder, hashtagsWithoutEndingWindowFile,\
         hashtagsWithEndingWindowFile, timeUnitWithOccurrencesFile, \
         hashtagsWithoutEndingWindowFile, hashtagsAllOccurrencesWithinWindowFile,\
         hashtagsWithoutEndingWindowWithoutLatticeApproximationFile
from datetime import datetime
from dateutil.relativedelta import relativedelta
from library.mrjobwrapper import runMRJob
from mr_analysis import MRAnalysis, PARAMS_DICT, START_TIME,\
                    END_TIME, WINDOW_OUTPUT_FOLDER
from library.file_io import FileIO

def iterateJsonFromFile(file):
    for data in FileIO.iterateJsonFromFile(file):
        if 'PARAMS_DICT' not in data: yield data

def getInputFiles(startTime, endTime, folderType='world'):
    current=startTime
    while current<=endTime:
        yield hdfsInputFolder%folderType+'%s_%s'%(current.year, current.month)
        current+=relativedelta(months=1)   

def mr_analysis(startTime, endTime, outputFolder, inputFilesStartTime=None, inputFilesEndTime=None):
    if not inputFilesStartTime: inputFilesStartTime=startTime; inputFilesEndTime=endTime
#    outputFile = hashtagsWithEndingWindowFile%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
#    outputFile = hashtagsWithoutEndingWindowFile%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
    outputFile = hashtagsWithoutEndingWindowWithoutLatticeApproximationFile%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
#    outputFile = hashtagsAllOccurrencesWithinWindowFile%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
#    outputFile = timeUnitWithOccurrencesFile%(outputFolder, startTime.strftime('%Y-%m-%d'), endTime.strftime('%Y-%m-%d'))
#    outputFile = 'mr_Data/timeUnitWithOccurrences'
    runMRJob(MRAnalysis, outputFile, getInputFiles(inputFilesStartTime, inputFilesEndTime), jobconf={'mapred.reduce.tasks':300})
    FileIO.writeToFileAsJson(PARAMS_DICT, outputFile)

if __name__ == '__main__':
#    inputFilesStartTime, inputFilesEndTime = None, None
    inputFilesStartTime, inputFilesEndTime = datetime(2011, 4, 1), datetime(2012, 1, 31)
    INPUT_START_TIME, INPUT_END_TIME = START_TIME, END_TIME 
    mr_analysis(INPUT_START_TIME, INPUT_END_TIME, WINDOW_OUTPUT_FOLDER, inputFilesStartTime=inputFilesStartTime, inputFilesEndTime=inputFilesEndTime)
