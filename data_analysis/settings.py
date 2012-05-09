'''
Created on May 8, 2012

@author: krishnakamath
'''
import os

hdfs_input_folder = 'hdfs:///user/kykamath/geo/hashtags/%s/'

fld_data_analysis = '/mnt/chevron/kykamath/data/geo/hashtags/data_analysis/%s_%s/%s/'
fld_sky_drive_data_analysis = os.path.expanduser('~/SkyDrive/Research/hashtags_and_geo/data_analysis/%s_%s/%s/') 
fld_sky_drive_data_analysis_images = fld_sky_drive_data_analysis+'images/'

f_tuo_normalized_occurrence_count_and_distribution_value = fld_data_analysis+'/tuo_normalized_occurrence_count_and_distribution_value'
f_tuo_lid_and_distribution_value = fld_data_analysis+'/tuo_lid_and_distribution_value'
f_tweet_count_stats = fld_data_analysis+'/tweet_count_stats'
f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage = fld_data_analysis+'/tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage'
f_tuo_rank_and_average_percentage_of_occurrences =  fld_data_analysis+'/tuo_rank_and_average_percentage_of_occurrences'

f_tuo_iid_and_tuo_is_peak_and_cumulative_percentage_of_occurrences =  fld_data_analysis+'/tuo_iid_and_tuo_is_peak_and_cumulative_percentage_of_occurrences'
