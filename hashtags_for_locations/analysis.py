'''
Created on Nov 19, 2011

@author: kykamath
'''
from settings import hdfsInputFolder, hashtagsWithoutEndingWindowFile, hashtagsWithEndingWindowFile
from datetime import datetime
from dateutil.relativedelta import relativedelta
from library.mrjobwrapper import runMRJob
from mr_analysis import MRAnalysis, PARAMS_DICT
from library.file_io import FileIO

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
#    startTime, endTime, outputFolder = datetime(2011, 4, 1), datetime(2012, 1, 31), 'complete' # Complete duration
#    startTime, endTime, outputFolder = datetime(2011, 5, 1), datetime(2011, 12, 31), 'complete_prop' # Complete propagation duration
    startTime, endTime, outputFolder = datetime(2011, 5, 1), datetime(2011, 10, 31), 'training' # Training duration
#    startTime, endTime, outputFolder = datetime(2011, 11, 1), datetime(2011, 12, 31), 'testing' # Testing duration
    
    mr_analysis(startTime, endTime, outputFolder)
