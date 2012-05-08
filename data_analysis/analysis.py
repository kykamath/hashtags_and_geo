'''
Created on May 7, 2012

@author: krishnakamath
'''
from library.file_io import FileIO
from dateutil.relativedelta import relativedelta
from library.mrjobwrapper import runMRJob
from mr_analysis import MRAnalysis, PARAMS_DICT, \
    MIN_HASHTAG_OCCURENCES
from datetime import datetime
from settings import hdfs_input_folder, \
    f_tuo_normalized_occurrence_count_and_distribution_value,\
    f_tweet_count_stats, f_tuo_lid_and_distribution_value, \
    f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus, \
    f_tuo_rank_and_average_percentage_of_occurrences

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

def mr_data_analysis(input_files_start_time, input_files_end_time, min_hashtag_occurrences):
#    output_file = f_tuo_normalized_occurrence_count_and_distribution_value%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)
#    output_file = f_tweet_count_stats%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)
#    output_file = f_tuo_lid_and_distribution_value%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)
    output_file = f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)
#    output_file = f_tuo_rank_and_average_percentage_of_occurrences%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)
    runMRJob(MRAnalysis, output_file, getInputFiles(input_files_start_time, input_files_end_time), jobconf={'mapred.reduce.tasks':300})
    FileIO.writeToFileAsJson(PARAMS_DICT, output_file)

if __name__ == '__main__':
#    input_files_start_time, input_files_end_time, min_hashtag_occurrences = datetime(2011, 2, 1), datetime(2011, 2, 27), 0
    input_files_start_time, input_files_end_time, min_hashtag_occurrences = datetime(2011, 2, 1), datetime(2012, 4, 30), MIN_HASHTAG_OCCURENCES
    mr_data_analysis(input_files_start_time, input_files_end_time, min_hashtag_occurrences)
