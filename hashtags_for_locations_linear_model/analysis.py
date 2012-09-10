'''
Created on Sept 9, 2012

@author: kykamath
'''
from dateutil.relativedelta import relativedelta
from library.file_io import FileIO
from library.mrjobwrapper import runMRJob
from datetime import datetime
from mr_analysis import MRTweetStats
from mr_analysis import PARAMS_DICT
from settings import f_tweet_stats
from settings import hdfs_input_folder
    
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
        
def mr_analysis(input_files_start_time, input_files_end_time):
    output_file = f_tweet_stats
    runMRJob(MRTweetStats,
             output_file,
             getInputFiles(input_files_start_time, input_files_end_time),
             jobconf={'mapred.reduce.tasks':300})
    FileIO.writeToFileAsJson(PARAMS_DICT, output_file)

if __name__ == '__main__':
    input_files_start_time, input_files_end_time = datetime(2011, 2, 1), datetime(2011, 7, 31)
    mr_analysis(input_files_start_time, input_files_end_time)
