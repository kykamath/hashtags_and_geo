'''
Created on Apr 13, 2012

@author: krishnakamath
'''
PARTIAL_WORLD_BOUNDARY = [[-58.447733,-153.457031], [72.127936,75.410156]]

hdfs_input_folder = 'hdfs:///user/kykamath/geo/hashtags/%s/'

train_location_objects_file = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/complete_prop/2011-05-01_2011-12-31/latticeGraph'


w_extra_hashtags_tag = 'w_extra_hashtags'
wout_extra_hashtags_tag = 'wout_extra_hashtags'
locations_influence_folder = '/mnt/chevron/kykamath/data/geo/hashtags/locations_influence/%s_%s/'
fld_locations_influence_test = '/mnt/chevron/kykamath/data/geo/hashtags/locations_influence/test/'
#locations_influence_folder = '/mnt/chevron/kykamath/data/geo/hashtags/locations_influence/%s_w_extra_hashtags/'

tuo_location_and_tuo_neighbor_location_and_pure_influence_score_file  = locations_influence_folder \
    + 'tuo_location_and_tuo_neighbor_location_and_pure_influence_score'
tuo_location_and_tuo_neighbor_location_and_influence_score_file  = locations_influence_folder \
    + 'tuo_location_and_tuo_neighbor_location_and_influence_score'
tuo_location_and_tuo_neighbor_location_and_mf_influence_type_and_similarity_file = locations_influence_folder \
    + 'tuo_location_and_tuo_neighbor_location_and_mf_influence_type_and_similarity'
tuo_location_and_tuo_neighbor_location_and_sharing_affinity_score_file = locations_influence_folder \
    + 'tuo_location_and_tuo_neighbor_location_and_sharing_affinity_score'
f_ltuo_hashtag_and_ltuo_location_and_pure_influence_score  = fld_locations_influence_test \
    + '%s/ltuo_hashtag_and_ltuo_location_and_pure_influence_score'

analysis_folder = locations_influence_folder%('analysis/', '')

hadoop_output_folder = '/mnt/chevron/kykamath/data/geo/hashtags/locations_influence/hadoop/%s/'
location_objects_file = hadoop_output_folder+'%s_%s/location_objects'
f_hashtag_objects = hadoop_output_folder+'%s_%s/hashtag_objects'
f_ltuo_location_and_ltuo_hashtag_and_occurrence_time = hadoop_output_folder+'%s_%s/ltuo_location_and_ltuo_hashtag_and_occurrence_time'
