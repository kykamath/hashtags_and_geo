'''
Created on Sep 26, 2012

@author: krishnakamath
'''
from datetime import datetime
from dateutil.relativedelta import relativedelta
from library.file_io import FileIO
from library.mrjobwrapper import runMRJob
from mr_analysis import HashtagsExtractor
from mr_analysis import PARAMS_DICT
from mr_analysis import WordObjectExtractor
from pprint import pprint
from settings import f_hashtags_extractor
from settings import f_hdfs_hashtags
from settings import f_word_objects_extractor
from settings import hdfs_input_folder
import time

class MRAnalysis():
    @staticmethod
    def run_job(mr_class, output_file, input_files_start_time, input_files_end_time):
        PARAMS_DICT['input_files_start_time'] = time.mktime(input_files_start_time.timetuple())
        PARAMS_DICT['input_files_end_time'] = time.mktime(input_files_end_time.timetuple())
        print 'Running map reduce with the following params:', pprint(PARAMS_DICT)
        runMRJob(
                     mr_class,
                     output_file,
                     MRAnalysis.get_input_files_with_tweets(input_files_start_time, input_files_end_time),
                     jobconf={'mapred.reduce.tasks':500}
                 )
        FileIO.writeToFileAsJson(PARAMS_DICT, output_file)
    @staticmethod
    def get_input_files_with_tweets(startTime, endTime, folderType='world'):
        current=startTime
        while current<=endTime:
            input_file = hdfs_input_folder%folderType+'%s_%s'%(current.year, current.month)
            print input_file
            yield input_file
            current+=relativedelta(months=1)
    @staticmethod
    def hashtags_extractor(input_files_start_time, input_files_end_time):
        mr_class = HashtagsExtractor
        output_file = f_hashtags_extractor
        MRAnalysis.run_job(mr_class, output_file, input_files_start_time, input_files_end_time)
    @staticmethod
    def word_object_extractor():
        mr_class = WordObjectExtractor
        output_file = f_word_objects_extractor
        runMRJob(
                     mr_class,
                     output_file,
                     [f_hdfs_hashtags],
                     jobconf={'mapred.reduce.tasks':500, 'mapred.task.timeout': 86400000}
                 )
    @staticmethod
    def run():
        input_files_start_time, input_files_end_time = \
                                datetime(2011, 2, 1), datetime(2012, 8, 31)
#        MRAnalysis.hashtags_extractor(input_files_start_time, input_files_end_time)
        MRAnalysis.word_object_extractor()

if __name__ == '__main__':
    MRAnalysis.run()
