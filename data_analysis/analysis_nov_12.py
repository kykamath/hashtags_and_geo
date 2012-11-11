'''
Created on Nov 9, 2012

@author: krishnakamath
'''
from datetime import datetime
from dateutil.relativedelta import relativedelta
from library.file_io import FileIO
from library.mrjobwrapper import runMRJob
from mr_analysis_nov_12 import DataStats
from mr_analysis_nov_12 import HashtagObjects
from mr_analysis_nov_12 import PARAMS_DICT
from pprint import pprint
from settings import hdfs_input_folder
from settings import f_data_stats
from settings import f_hashtag_objects
import time

class MRAnalysis():
    @staticmethod
    def get_input_files_with_tweets(startTime, endTime, folderType='world'):
        current=startTime
        while current<=endTime:
            input_file = hdfs_input_folder%folderType+'%s_%s'%(current.year, current.month)
            print input_file
            yield input_file
            current+=relativedelta(months=1)
    @staticmethod
    def run_job(mr_class, output_file, input_files_start_time, input_files_end_time):
        PARAMS_DICT['input_files_start_time'] = time.mktime(input_files_start_time.timetuple())
        PARAMS_DICT['input_files_end_time'] = time.mktime(input_files_end_time.timetuple())
        print 'Running map reduce with the following params:', pprint(PARAMS_DICT)
        runMRJob(mr_class,
                 output_file,
                 MRAnalysis.get_input_files_with_tweets(input_files_start_time, input_files_end_time),
                 jobconf={'mapred.reduce.tasks':500})
        FileIO.writeToFileAsJson(PARAMS_DICT, output_file)
    @staticmethod
    def data_stats(input_files_start_time, input_files_end_time):
        mr_class = DataStats
        output_file = f_data_stats
        MRAnalysis.run_job(mr_class, output_file, input_files_start_time, input_files_end_time)
    @staticmethod
    def hashtag_objects(input_files_start_time, input_files_end_time):
        mr_class = HashtagObjects
        output_file = f_hashtag_objects
        MRAnalysis.run_job(mr_class, output_file, input_files_start_time, input_files_end_time)
    @staticmethod
    def run():
        input_files_start_time, input_files_end_time = datetime(2011, 2, 1), datetime(2012, 10, 31)
#        MRAnalysis.data_stats(input_files_start_time, input_files_end_time)
        MRAnalysis.hashtag_objects(input_files_start_time, input_files_end_time)
            
if __name__ == '__main__':
    MRAnalysis.run()
