'''
Created on May 8, 2012

@author: krishnakamath
'''
hdfs_input_folder = 'hdfs:///user/kykamath/geo/hashtags/%s/'

fld_data_analysis = '/mnt/chevron/kykamath/data/geo/hashtags/hashtag_propagation_in_groups/%s/'

f_hdfs_hashtags = hdfs_input_folder%'2011_2_to_2012_8'+'/min_num_of_hashtags_50_with_words/hashtags'

f_hashtags_extractor = fld_data_analysis%'hashtags_extractor'+'hashtags'
f_word_objects_extractor = fld_data_analysis%'word_objects_extractor'+ 'word_objects'
    