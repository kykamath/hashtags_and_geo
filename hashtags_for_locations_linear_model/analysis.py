'''
Created on Sept 9, 2012

@author: kykamath
'''
from dateutil.relativedelta import relativedelta
from library.file_io import FileIO
from library.mrjobwrapper import runMRJob
from library.mrjobwrapper import runMRJobAndYieldResult
from datetime import datetime
from mr_analysis import HashtagsByUTMId
from mr_analysis import HashtagsExtractor
from mr_analysis import HastagsWithUTMIdObject
from mr_analysis import PARAMS_DICT
from mr_analysis import TweetStats
from settings import f_hashtags_by_utm_id
from settings import f_hashtags_extractor
from settings import f_hashtags_with_utm_id_object
from settings import f_tweet_stats
from settings import hdfs_input_folder

def getInputFiles(startTime, endTime, folderType='world'):
    current=startTime
    while current<=endTime:
        input_file = hdfs_input_folder%folderType+'%s_%s'%(current.year,
                                                           current.month)
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
                 mrJobClassParams = {'job_id': 'as'},
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
    @staticmethod
    def hashtags_by_utm_id(input_files_start_time, input_files_end_time):
        mr_class = HashtagsByUTMId
        output_file = f_hashtags_by_utm_id
        runMRJob(mr_class,
                 output_file,
                 getInputFiles(input_files_start_time, input_files_end_time),
                 jobconf={'mapred.reduce.tasks':500})
        FileIO.writeToFileAsJson(PARAMS_DICT, output_file)
    @staticmethod
    def hashtags_by_valid_utm_id(input_files_start_time, input_files_end_time):
        mr_class = HashtagsByUTMId
        result = runMRJobAndYieldResult(
                    mr_class,
                    getInputFiles(input_files_start_time,
                                  input_files_end_time),
                    mrJobClassParams = {'job_id': 
                                     HashtagsByUTMId.JOBS_TO_GET_VALID_UTM_IDS},
                    jobconf={'mapred.reduce.tasks':500}
                )
        for r in result:
            print r
    @staticmethod
    def hashtags_with_utm_id_object(input_files_start_time,
                                    input_files_end_time):
        mr_class = HastagsWithUTMIdObject
        output_file = f_hashtags_with_utm_id_object
        runMRJob(mr_class,
                 output_file,
                 getInputFiles(input_files_start_time, input_files_end_time),
                 jobconf={'mapred.reduce.tasks':500})
        FileIO.writeToFileAsJson(PARAMS_DICT, output_file)    
    
    @staticmethod
    def run():
        input_files_start_time, input_files_end_time = \
                        datetime(2011, 2, 1), datetime(2011, 4, 30)
    #    input_files_start_time, input_files_end_time = \
    #                            datetime(2011, 2, 1), datetime(2012, 8, 31)
    #    MRAnalysis.tweet_stats(input_files_start_time, input_files_end_time)
    #    MRAnalysis.hashtags_extractor(input_files_start_time, input_files_end_time)
        MRAnalysis.hashtags_by_valid_utm_id(input_files_start_time,
                                            input_files_end_time)
    #    MRAnalysis.hashtags_by_utm_id(input_files_start_time, input_files_end_time)
    #    MRAnalysis.hashtags_with_utm_id_object(input_files_start_time,
    #                                           input_files_end_time)
    
class GeneralAnalysis(object):
    @staticmethod
    def print_valid_utm_ids():
        ''' Prints list of valid utm_ids.
        '''
        print [utm_object['utm_id'] 
               for utm_object in FileIO.iterateJsonFromFile(
                                                    f_hashtags_by_utm_id,
                                                    remove_params_dict=True)]
    @staticmethod
    def run():
        GeneralAnalysis.print_valid_utm_ids()

if __name__ == '__main__':
#    MRAnalysis.run()
    GeneralAnalysis.run()
    
