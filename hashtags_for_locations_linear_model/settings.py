'''
Created on Sept 9, 2012

@author: kykamath
'''
import os

hdfs_input_folder = 'hdfs:///user/kykamath/geo/hashtags/%s/'
analysis_folder = '/mnt/chevron/kykamath/data/geo/hashtags/'\
                    'hashtags_for_locations_linear_model/%s/'
                    
# Analysis files.
f_tweet_stats = analysis_folder%'tweet_stats'+'stats'
f_hashtags_extractor = analysis_folder%'hashtags_extractor'+'hashtags'
f_hashtag_dist_by_accuracy = analysis_folder%'hashtag_dist_by_accuracy'+\
                                                    'hashtag_dist'
f_hashtags_by_utm_id = analysis_folder%'hashtags_by_utm_id'+'hashtags_by_utm_id'
f_hashtags_with_utm_id_object = analysis_folder%'hashtags_with_utm_id_object'+\
                                                'hashtags_with_utm_id_object'

fld_google_drive_data_analysis = os.path.expanduser('~/Google Drive/Desktop/'\
            'hashtags_and_geo/hashtags_for_locations_linear_model/%s') 

# Bounding boxes
NY_BB = [[40.574326,-74.045311], [40.939452,-73.728081]]
