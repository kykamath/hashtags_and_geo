from library.file_io import FileIO

for data in FileIO.iterateJsonFromFile('/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/testing/models_1/2011-09-01_2011-11-01/360_900/100/linear_regression'):
    for loc in data['mf_location_to_ideal_hashtags_rank']:
        print data['tu'], loc, data['mf_location_to_ideal_hashtags_rank'][loc]
    exit()
    
#import numpy as np
#from operator import itemgetter
##def get_occurrences_stats(occurrences1, occurrences2):
##        no_of_occurrences_after_appearing_in_location, no_of_occurrences_before_appearing_in_location = 0., 0.
##        occurrences1=sorted(occurrences1)
##        occurrences2=sorted(occurrences2)
##        no_of_total_occurrences_between_location_pair = len(occurrences1)*len(occurrences2)*1.
##        for occurrence1 in occurrences1:
##            for occurrence2 in occurrences2:
##                if occurrence1<occurrence2: no_of_occurrences_after_appearing_in_location+=1
##                elif occurrence1>occurrence2: no_of_occurrences_before_appearing_in_location+=1
##        return no_of_occurrences_after_appearing_in_location, no_of_occurrences_before_appearing_in_location, no_of_total_occurrences_between_location_pair
##
##def get_influence_scores(location_occurrences, neighbor_location_occurrences):
##        (no_of_occurrences_after_appearing_in_location, \
##         no_of_occurrences_before_appearing_in_location, \
##         no_of_total_occurrences_between_location_pair) =\
##            get_occurrences_stats(location_occurrences, neighbor_location_occurrences)
##        total_nof_occurrences = float(len(location_occurrences) + len(neighbor_location_occurrences))
##        ratio_of_occurrences_in_location = len(location_occurrences)/total_nof_occurrences
##        ratio_of_occurrences_in_neighbor_location = len(neighbor_location_occurrences)/total_nof_occurrences
##        return ( ratio_of_occurrences_in_location*no_of_occurrences_after_appearing_in_location - ratio_of_occurrences_in_neighbor_location*no_of_occurrences_before_appearing_in_location) / no_of_total_occurrences_between_location_pair
##    
##for location_object in FileIO.iterateJsonFromFile('data/40.6000_-73.9500'):
##    tuples_of_neighbor_location_and_pure_influence_score = []
##    location_hashtag_set = set(location_object['hashtags'])
##    for neighbor_location, map_from_hashtag_to_tuples_of_occurrences_and_time_range in location_object['links'].iteritems():
##        if neighbor_location in ['50.7500_2.1750', '35.5250_-5.8000', '27.5500_-16.6750']:
##            pure_influence_scores = []
###            ht_n_score = []
##            for hashtag, (neighbor_location_occurrences, time_range) in map_from_hashtag_to_tuples_of_occurrences_and_time_range.iteritems():
##                if hashtag in location_object['hashtags']:
##                    location_occurrences = location_object['hashtags'][hashtag][0]
##                    pure_influence_scores.append(get_influence_scores(location_occurrences, neighbor_location_occurrences))
###                    ht_n_score.append((hashtag, get_influence_scores(location_occurrences, neighbor_location_occurrences)))
##            neighbor_location_hashtag_set = set(map_from_hashtag_to_tuples_of_occurrences_and_time_range.keys())
##            for hashtag in location_hashtag_set.difference(neighbor_location_hashtag_set): pure_influence_scores.append(1.0)
##            for hashtag in neighbor_location_hashtag_set.difference(location_hashtag_set): pure_influence_scores.append(-1.0)
##            mean_pure_influence_score = np.mean(pure_influence_scores)
##            tuples_of_neighbor_location_and_pure_influence_score.append([neighbor_location, mean_pure_influence_score])
##        tuples_of_neighbor_location_and_pure_influence_score= sorted(tuples_of_neighbor_location_and_pure_influence_score, key=itemgetter(1))
##    print tuples_of_neighbor_location_and_pure_influence_score
#
#
#from collections import defaultdict
#
#def get_feature_vectors(data):
#    mf_model_id_to_mf_location_to_hashtags_ranked_by_model =\
#                                                         data['mf_model_id_to_mf_location_to_hashtags_ranked_by_model']
#    mf_location_to_ideal_hashtags_rank = data['mf_location_to_ideal_hashtags_rank']
#    conf = data['conf']
#    tu = data['tu']
#    
#    mf_to_location_to_mf_hashtag_to_mf_model_id_to_score = {}
#    for model_id, mf_location_to_hashtags_ranked_by_model in \
#            mf_model_id_to_mf_location_to_hashtags_ranked_by_model.iteritems():
#        for location, hashtags_ranked_by_model in mf_location_to_hashtags_ranked_by_model.iteritems():
#<<<<<<< HEAD
#            for hashtag, score in hashtags_ranked_by_model:
#                if location not in mf_to_location_to_mf_hashtag_to_mf_model_id_to_score:
#                    mf_to_location_to_mf_hashtag_to_mf_model_id_to_score[location] = defaultdict(dict)
#                mf_to_location_to_mf_hashtag_to_mf_model_id_to_score[location][location][model_id] = score
##                print model_id, location, hashtag, score
#
#    print 'x'
#        
##    for location, ltuo_hashtag_and_perct in mf_location_to_ideal_hashtags_rank.iteritems():
##        for hashtag, perct in ltuo_hashtag_and_perct:
##            print location, hashtag, perct
#=======
##            print model_id, location, hashtags_ranked_by_model
#            for hashtag, score in hashtags_ranked_by_model:
#                mf_hashtag_to_mf_model_id_to_score[hashtag][model_id] = score
#    
#    for location, ltuo_hashtag_and_perct in mf_location_to_ideal_hashtags_rank.iteritems():
#        for hashtag, perct in ltuo_hashtag_and_perct:
#            if hashtag in mf_hashtag_to_mf_model_id_to_score:
#                mf_hashtag_to_mf_model_id_to_score[hashtag]['value_to_predict'] = perct
#                yield location, mf_hashtag_to_mf_model_id_to_score[hashtag]
#
#for data in FileIO.iterateJsonFromFile('data/linear_regression'):
#    for fv in get_feature_vectors(data):
#        print fv
#>>>>>>> branch 'master' of ssh://git@github.com/kykamath/hashtags_and_geo.git
##            mf_model_id_to_score = {}
##            for model_id, mf_location_to_hashtags_ranked_by_model in \
##                    mf_model_id_to_mf_location_to_hashtags_ranked_by_model.iteritems():
#                
#                
#                
#                
