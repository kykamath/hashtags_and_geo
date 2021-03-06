'''
Created on May 8, 2012

@author: krishnakamath
'''
import os

hdfs_input_folder = 'hdfs:///user/kykamath/geo/hashtags/%s/'

fld_data_analysis = '/mnt/chevron/kykamath/data/geo/hashtags/data_analysis/%s_%s/%s/'
fld_sky_drive_data_analysis = os.path.expanduser('~/SkyDrive/Research/hashtags_and_geo/data_analysis/%s_%s/%s/') 
fld_sky_drive_data_analysis_images = fld_sky_drive_data_analysis+'images/'

f_hashtag_objects = fld_data_analysis+'/hashtag_objects'
f_tuo_normalized_occurrence_count_and_distribution_value = fld_data_analysis+'/tuo_normalized_occurrence_count_and_distribution_value'
f_tuo_lid_and_distribution_value = fld_data_analysis+'/tuo_lid_and_distribution_value'
f_tweet_count_stats = fld_data_analysis+'/tweet_count_stats'
f_tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak = \
    fld_data_analysis+'/tuo_hashtag_and_occurrence_count_and_entropy_and_focus_and_coverage_and_peak'
f_tuo_rank_and_average_percentage_of_occurrences =  fld_data_analysis+'/tuo_rank_and_average_percentage_of_occurrences'
#interval_stats = [peak, percentage_of_occurrences, cumulative_percentage_of_occurrences]
f_tuo_iid_and_interval_stats = fld_data_analysis+'/tuo_iid_and_interval_stats'
f_tuo_normalized_iid_and_tuo_prct_of_occurrences_and_entropy_and_focus_and_coverage = \
    fld_data_analysis+'/tuo_normalized_iid_and_tuo_prct_of_occurrences_and_entropy_and_focus_and_coverage'
f_tuo_lid_and_ltuo_other_lid_and_temporal_distance = \
    fld_data_analysis+'/tuo_lid_and_ltuo_other_lid_and_temporal_distance'
f_tuo_lid_and_ltuo_other_lid_and_no_of_co_occurrences = fld_data_analysis+'/tuo_lid_and_ltuo_other_lid_and_no_of_co_occurrences'
f_tuo_high_accuracy_lid_and_distribution = fld_data_analysis+'/tuo_high_accuracy_lid_and_distribution'
f_tuo_no_of_hashtags_and_count = fld_data_analysis+'/tuo_no_of_hashtags_and_count'
f_tuo_no_of_locations_and_count = fld_data_analysis+'/tuo_no_of_locations_and_count'
f_tuo_iid_and_perct_change_of_occurrences = fld_data_analysis+'/tuo_iid_and_perct_change_of_occurrences'
f_tuo_no_of_peak_lids_and_count = fld_data_analysis + '/tuo_no_of_peak_lids_and_count'
f_tuo_valid_focus_lid_pair_and_common_hashtag_affinity_score = fld_data_analysis + '/tuo_valid_focus_lid_pair_and_common_hashtag_affinity_score'
f_tuo_valid_focus_lid_pair_and_temporal_affinity_score = fld_data_analysis + '/tuo_valid_focus_lid_pair_and_temporal_affinity_score'
#f_tuo_lid_and_ltuo_other_lid_and_no_o = fld_data_analysis+'/tuo_iid_and_perct_change_of_occurrences'

# Nov 12 analysis
fld_data_analysis = '/mnt/chevron/kykamath/data/geo/hashtags/data_analysis/'
fld_data_analysis_results = os.path.expanduser('~/Google Drive/Desktop/hashtags_and_geo/data_analysis/%s') 
f_hashtag_objects_on_dfs = 'hdfs:///user/kykamath/geo/hashtags/2011_2_to_2012_10/min_num_of_hashtags_50/hashtags'
f_data_stats = fld_data_analysis+'/data_stats'
f_dense_data_stats = fld_data_analysis+'/dense_data_stats'
f_example_for_caverlee = fld_data_analysis+'/example_for_caverlee' 
f_generate_data_for_public = fld_data_analysis+'/hashtag_occurrence_data' 
f_hashtag_objects = fld_data_analysis+'/hashtag_objects'
f_hashtag_and_location_distribution = fld_data_analysis+'/hashtag_and_location_distribution'
f_dense_hashtag_distribution_in_locations = fld_data_analysis+'/dense_hashtag_distribution_in_locations'
f_dense_hashtags_similarity_and_lag = fld_data_analysis+'/dense_hashtags_similarity_and_lag'
f_hashtag_spatial_metrics = fld_data_analysis+'/hashtag_spatial_metrics'
f_iid_spatial_metrics = fld_data_analysis+'/iid_spatial_metrics'
f_norm_iid_spatial_metrics = fld_data_analysis+'/norm_iid_spatial_metrics'
