'''
Created on Sept 9, 2012

@author: kykamath
'''
from dateutil.relativedelta import relativedelta
from library.file_io import FileIO
from library.mrjobwrapper import runMRJob
from datetime import datetime
from mr_analysis import HashtagsByUTMId
from mr_analysis import HashtagsExtractor
from mr_analysis import PARAMS_DICT
from mr_analysis import TweetStats
from settings import f_hashtags_by_utm_id
from settings import f_hashtags_extractor
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

class MRAnalysis(object):
    @staticmethod
    def tweet_stats(input_files_start_time, input_files_end_time):
        mr_class = TweetStats
        output_file = f_tweet_stats
        runMRJob(mr_class,
                 output_file,
                 getInputFiles(input_files_start_time, input_files_end_time),
                 jobconf={'mapred.reduce.tasks':300})
        FileIO.writeToFileAsJson(PARAMS_DICT, output_file)
    @staticmethod
    def hashtags_by_utm_id(input_files_start_time, input_files_end_time):
        mr_class = HashtagsByUTMId
        output_file = f_hashtags_by_utm_id
        runMRJob(mr_class,
                 output_file,
                 getInputFiles(input_files_start_time, input_files_end_time),
                 jobconf={'mapred.reduce.tasks':300})
        FileIO.writeToFileAsJson(PARAMS_DICT, output_file)
    @staticmethod
    def hashtags_extractor(input_files_start_time, input_files_end_time):
        mr_class = HashtagsExtractor
        output_file = f_hashtags_extractor
        runMRJob(mr_class,
                 output_file,
                 getInputFiles(input_files_start_time, input_files_end_time),
                 jobconf={'mapred.reduce.tasks':500})
        FileIO.writeToFileAsJson(PARAMS_DICT, output_file)

if __name__ == '__main__':
#    input_files_start_time, input_files_end_time = \
#                            datetime(2011, 2, 1), datetime(2011, 4, 30)
    input_files_start_time, input_files_end_time = \
                            datetime(2011, 2, 1), datetime(2012, 8, 31)
#    MRAnalysis.tweet_stats(input_files_start_time, input_files_end_time)
    MRAnalysis.hashtags_extractor(input_files_start_time, input_files_end_time)
#    MRAnalysis.hashtags_by_utm_id(input_files_start_time, input_files_end_time)
