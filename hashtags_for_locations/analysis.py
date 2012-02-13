'''
Created on Nov 19, 2011

@author: kykamath
'''
from settings import hdfsInputFolder, hashtagsWithoutEndingWindowFile, hashtagsWithEndingWindowFile
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

def mr_analysis(startTime, endTime, outputFolder):
    outputFile = hashtagsWithEndingWindowFile%outputFolder
    runMRJob(MRAnalysis, outputFile, getInputFiles(startTime, endTime), jobconf={'mapred.reduce.tasks':300})
    FileIO.writeToFileAsJson(PARAMS_DICT, outputFile)

if __name__ == '__main__':
    mr_analysis(START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER)
