'''
Created on Apr 13, 2012

@author: krishnakamath
'''

hdfs_input_folder = 'hdfs:///user/kykamath/geo/hashtags/%s/'

train_location_objects_file = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/complete_prop/2011-05-01_2011-12-31/latticeGraph'


locations_influence_folder = '/mnt/chevron/kykamath/data/geo/hashtags/locations_influence/%s/'
#locations_influence_folder = '/mnt/chevron/kykamath/data/geo/hashtags/locations_influence/%s_w_extra_hashtags/'

tuo_location_and_tuo_neighbor_location_and_pure_influence_score_file  = locations_influence_folder \
    + 'tuo_location_and_tuo_neighbor_location_and_pure_influence_score'
tuo_location_and_tuo_neighbor_location_and_influence_score_file  = locations_influence_folder \
    + 'tuo_location_and_tuo_neighbor_location_and_influence_score'

analysis_folder = locations_influence_folder%'analysis/'

hadoop_output_folder = locations_influence_folder%'hadoop'+'/%s/'
location_objects_file = hadoop_output_folder+'%s_%s/location_objects'
