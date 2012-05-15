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
    f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak, \
    f_tuo_rank_and_average_percentage_of_occurrences, \
    f_tuo_iid_and_interval_stats, \
    f_tuo_normalized_iid_and_tuo_prct_of_occurrences_and_entropy_and_focus_and_coverage,\
    f_hashtag_objects, f_tuo_lid_and_ltuo_other_lid_and_temporal_distance, \
    f_tuo_lid_and_ltuo_other_lid_and_no_of_co_occurrences, \
    f_tuo_high_accuracy_lid_and_distribution, f_tuo_no_of_hashtags_and_count,\
    f_tuo_no_of_locations_and_count, f_tuo_iid_and_perct_change_of_occurrences,\
    f_tuo_no_of_peak_lids_and_count
    
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
        
def getPreprocessedHashtagsFile():
    input_file = 'hdfs:///user/kykamath/geo/hashtags/2011-02-01_2012-04-30/50/hashtag_objects'
    print input_file
    yield input_file

def mr_data_analysis(input_files_start_time, input_files_end_time, min_hashtag_occurrences):
#    output_file = f_tuo_normalized_occurrence_count_and_distribution_value%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)
#    output_file = f_tweet_count_stats%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)
#    output_file = f_tuo_lid_and_distribution_value%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)
#    output_file = f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)
#    output_file = f_tuo_rank_and_average_percentage_of_occurrences%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)

#    output_file = f_tuo_iid_and_interval_stats%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)
#    output_file = f_tuo_iid_and_perct_change_of_occurrences%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)
#    output_file = f_tuo_normalized_iid_and_tuo_prct_of_occurrences_and_entropy_and_focus_and_coverage%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)

#    output_file = f_hashtag_objects%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)

#    output_file = f_tuo_lid_and_ltuo_other_lid_and_temporal_distance%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)
    output_file = f_tuo_lid_and_ltuo_other_lid_and_no_of_co_occurrences%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)
#    output_file = f_tuo_high_accuracy_lid_and_distribution%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)
#    output_file = f_tuo_no_of_hashtags_and_count%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)
#    output_file = f_tuo_no_of_locations_and_count%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)

#    output_file = f_tuo_no_of_peak_lids_and_count%(input_files_start_time.strftime('%Y-%m-%d'), input_files_end_time.strftime('%Y-%m-%d'), min_hashtag_occurrences)

    print PARAMS_DICT
#    runMRJob(MRAnalysis, output_file, getInputFiles(input_files_start_time, input_files_end_time), jobconf={'mapred.reduce.tasks':300})
    runMRJob(MRAnalysis, output_file, getPreprocessedHashtagsFile(), jobconf={'mapred.reduce.tasks':300})
    FileIO.writeToFileAsJson(PARAMS_DICT, output_file)

if __name__ == '__main__':
#    input_files_start_time, input_files_end_time, min_hashtag_occurrences = datetime(2011, 2, 1), datetime(2011, 2, 27), 0
    input_files_start_time, input_files_end_time, min_hashtag_occurrences = datetime(2011, 2, 1), datetime(2012, 4, 30), MIN_HASHTAG_OCCURENCES
#    input_files_start_time, input_files_end_time, min_hashtag_occurrences = datetime(2011, 2, 1), datetime(2011, 5, 30), MIN_HASHTAG_OCCURENCES
    mr_data_analysis(input_files_start_time, input_files_end_time, min_hashtag_occurrences)
