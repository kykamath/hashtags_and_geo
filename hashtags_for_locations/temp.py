#import matplotlib.pyplot as plt
#import numpy as np
#from scipy.stats import gaussian_kde
##data = [1.5]*7 + [2.5]*2 + [3.5]*8 + [4.5]*3 + [5.5]*1 + [6.5]*8
#data = [1.5]*7 + [5]*2 + [3.5]*8 + [7.5]*3
#density = gaussian_kde(data)
#xs = np.linspace(0,8,200)
#density.covariance_factor = lambda : .25
#density._compute_covariance()
#plt.plot(xs,density(xs), c='y')
#plt.fill_between(xs,density(xs),0,color='r')
##plt.hist(data)
#plt.show()

#from itertools import combinations
#
#print list(combinations(range(20),2))

import random
import numpy as np
#
#observed_data = [-1,1,1,-1,-1,-1,-1,-1,-1]
#
#iteration_data = [random.sample([-1,1],1)[0] for i in range(len(observed_data))]
#print actual_data
#print observed_data, np.mean()

#samples = [[54, 51, 58, 44, 55, 52, 42, 47, 58, 46], [54, 73, 53, 70, 73, 68, 52, 65, 65]]
class MonteCarloSimulation(object):
    '''
    Part of this code was got from the implementation in the book "Statistics is Easy!" By Dennis Shasha and
    Manda Wilson
    '''
    NUM_OF_SIMULATIONS = 10000
    @staticmethod
    def _shuffle(grps):
        num_grps = len(grps)
        pool = []
        # pool all values
        for i in range(num_grps):
            pool.extend(grps[i])
        # mix them up
        random.shuffle(pool)
        # reassign to groups of same size as original groups
        new_grps = []
        start_index = 0
        end_index = 0
        for i in range(num_grps):
            end_index = start_index + len(grps[i])
            new_grps.append(pool[start_index:end_index])
            start_index = end_index
        return new_grps
    @staticmethod
    # subtracts group a mean from group b mean and returns result
    def _meandiff(grpA, grpB):
        return sum(grpB) / float(len(grpB)) - sum(grpA) / float(len(grpA))

    @staticmethod
    def probability_of_data_extracted_from_same_sample(sample1, sample2):
        ''' Difference between Two Means Significance Test
        '''
        samples = [sample1, sample2] 
        a, b = 0, 1
        observed_mean_diff = MonteCarloSimulation._meandiff(samples[a], samples[b])
        count = 0
        num_shuffles = MonteCarloSimulation.NUM_OF_SIMULATIONS
        for i in range(num_shuffles):
            new_samples = MonteCarloSimulation._shuffle(samples)
            mean_diff = MonteCarloSimulation._meandiff(new_samples[a], new_samples[b])
            # if the observed difference is negative, look for differences that are smaller
            # if the observed difference is positive, look for differences that are greater
            if observed_mean_diff < 0 and mean_diff <= observed_mean_diff: count = count + 1
            elif observed_mean_diff >= 0 and mean_diff >= observed_mean_diff: count = count + 1
        return (count / float(num_shuffles))

observed_data = [54, 51, 58, 44, 55, 52, 42, 47, 58, 46]
expected_data = [54, 73, 53, 70, 73, 68, 52, 65, 65]

#observed_data = [0,0,0,0,0,0,0,0,0]
###observed_data = [0,1,1,0,1,0,1,0,1]
###observed_data = [random.sample([0,1],1)[0] for i in range(len(observed_data))]
#expected_data = [random.sample([0,1],1)[0] for i in range(len(observed_data))]

print observed_data, expected_data
print MonteCarloSimulation.probability_of_data_extracted_from_same_sample(observed_data, expected_data)

#######################################
##
## Output
##
#######################################
#
#print "Observed difference of two means: %.2f" % observed_mean_diff 
#print count, "out of", num_shuffles, "experiments had a difference of two means",
#if observed_mean_diff < 0:
#    print "less than or equal to",
#else:
#    print "greater than or equal to",
#print "%.2f" % observed_mean_diff, "."
#print "The chance of getting a difference of two means",
#if observed_mean_diff < 0:
#    print "less than or equal to",
#else:
#    print "greater than or equal to",
#print "%.2f" % observed_mean_diff, "is", (count / float(num_shuffles)), "."

##!/usr/bin/env python
#"""
#See pcolor_demo2 for a much faster way of generating pcolor plots
#"""
#from __future__ import division
#from pylab import *
#
#def func3(x,y):
#    return (1- x/2 + x**5 + y**3)*exp(-x**2-y**2)
#
#
## make these smaller to increase the resolution
#dx, dy = 0.05, 0.05
#
#x = arange(-3.0, 3.0, dx)
#y = arange(-3.0, 3.0, dy)
#X,Y = meshgrid(x, y)
#
##Z = func3(X, Y)
#
#Z = [[1,2,3],[1,2,3],[1,3,3]]
#
#
#ax = subplot(111)
#im = imshow(Z, cmap=cm.jet)
##im.set_interpolation('nearest')
##im.set_interpolation('bicubic')
#im.set_interpolation('bilinear')
##ax.set_image_extent(-3, 3, -3, 3)
#
#show()
#gap_perct = 0.1
#occ_times_at_gap_perct = range(10)
#ltuo_perct_and_occ_time = [(int((gap_perct*i+gap_perct)*100), j)for i, j in enumerate(occ_times_at_gap_perct)]
#for perct1, occ_time1 in ltuo_perct_and_occ_time:
#    for perct2, occ_time2 in ltuo_perct_and_occ_time:
#        print perct1, perct2, max(occ_time2-occ_time1, 0.0)
#for i, val in enumerate(l):
#    for 

#import numpy as np
#from operator import itemgetter
#
#def get_items_at_gap(input_list, gap_perct):
#    return map(
#                  lambda index: input_list[index],
#                  map(lambda index: int(index)-1, np.arange(gap_perct,1+gap_perct,gap_perct)*100)
#              )
#
#print get_items_at_gap(range(100), 0.2)
#print map(lambda index: int(index)-1, np.arange(0.02,1+0.02,0.02)*100)
#print input_list[1]

#print map(itemgetter(map(int, np.arange(0,1,0.02)*100)), input_list)

#for perct in xrange()
#list_len = len(input_list)
#print input_list[int(perct*list_len)]


#def m(l):
#    l[0] = -1
#    
#def m2(l):
#    l = [-1 for i in range(10)]
#
#l = list([i for i in range(10)])
#a=[1,2,3]
#b=list(a)
#print a, b
#a[0]=-1
#print a, b
##print l
##m(l)
##print l
##m2(l)
##print l
##from library.file_io import FileIO
##
##for data in FileIO.iterateJsonFromFile('linear_regression_small'):
###    print data.leys()
##    for loc in data['mf_location_to_ideal_hashtags_rank']:
##        print data['tu'], loc, sum(zip(*data['mf_location_to_ideal_hashtags_rank'][loc])[1])
##    exit()
##    
###import numpy as np
###from operator import itemgetter
####def get_occurrences_stats(occurrences1, occurrences2):
####        no_of_occurrences_after_appearing_in_location, no_of_occurrences_before_appearing_in_location = 0., 0.
####        occurrences1=sorted(occurrences1)
####        occurrences2=sorted(occurrences2)
####        no_of_total_occurrences_between_location_pair = len(occurrences1)*len(occurrences2)*1.
####        for occurrence1 in occurrences1:
####            for occurrence2 in occurrences2:
####                if occurrence1<occurrence2: no_of_occurrences_after_appearing_in_location+=1
####                elif occurrence1>occurrence2: no_of_occurrences_before_appearing_in_location+=1
####        return no_of_occurrences_after_appearing_in_location, no_of_occurrences_before_appearing_in_location, no_of_total_occurrences_between_location_pair
####
####def get_influence_scores(location_occurrences, neighbor_location_occurrences):
####        (no_of_occurrences_after_appearing_in_location, \
####         no_of_occurrences_before_appearing_in_location, \
####         no_of_total_occurrences_between_location_pair) =\
####            get_occurrences_stats(location_occurrences, neighbor_location_occurrences)
####        total_nof_occurrences = float(len(location_occurrences) + len(neighbor_location_occurrences))
####        ratio_of_occurrences_in_location = len(location_occurrences)/total_nof_occurrences
####        ratio_of_occurrences_in_neighbor_location = len(neighbor_location_occurrences)/total_nof_occurrences
####        return ( ratio_of_occurrences_in_location*no_of_occurrences_after_appearing_in_location - ratio_of_occurrences_in_neighbor_location*no_of_occurrences_before_appearing_in_location) / no_of_total_occurrences_between_location_pair
####    
####for location_object in FileIO.iterateJsonFromFile('data/40.6000_-73.9500'):
####    tuples_of_neighbor_location_and_pure_influence_score = []
####    location_hashtag_set = set(location_object['hashtags'])
####    for neighbor_location, map_from_hashtag_to_tuples_of_occurrences_and_time_range in location_object['links'].iteritems():
####        if neighbor_location in ['50.7500_2.1750', '35.5250_-5.8000', '27.5500_-16.6750']:
####            pure_influence_scores = []
#####            ht_n_score = []
####            for hashtag, (neighbor_location_occurrences, time_range) in map_from_hashtag_to_tuples_of_occurrences_and_time_range.iteritems():
####                if hashtag in location_object['hashtags']:
####                    location_occurrences = location_object['hashtags'][hashtag][0]
####                    pure_influence_scores.append(get_influence_scores(location_occurrences, neighbor_location_occurrences))
#####                    ht_n_score.append((hashtag, get_influence_scores(location_occurrences, neighbor_location_occurrences)))
####            neighbor_location_hashtag_set = set(map_from_hashtag_to_tuples_of_occurrences_and_time_range.keys())
####            for hashtag in location_hashtag_set.difference(neighbor_location_hashtag_set): pure_influence_scores.append(1.0)
####            for hashtag in neighbor_location_hashtag_set.difference(location_hashtag_set): pure_influence_scores.append(-1.0)
####            mean_pure_influence_score = np.mean(pure_influence_scores)
####            tuples_of_neighbor_location_and_pure_influence_score.append([neighbor_location, mean_pure_influence_score])
####        tuples_of_neighbor_location_and_pure_influence_score= sorted(tuples_of_neighbor_location_and_pure_influence_score, key=itemgetter(1))
####    print tuples_of_neighbor_location_and_pure_influence_score
###
###
###from collections import defaultdict
###
###def get_feature_vectors(data):
###    mf_model_id_to_mf_location_to_hashtags_ranked_by_model =\
###                                                         data['mf_model_id_to_mf_location_to_hashtags_ranked_by_model']
###    mf_location_to_ideal_hashtags_rank = data['mf_location_to_ideal_hashtags_rank']
###    conf = data['conf']
###    tu = data['tu']
###    
###    mf_to_location_to_mf_hashtag_to_mf_model_id_to_score = {}
###    for model_id, mf_location_to_hashtags_ranked_by_model in \
###            mf_model_id_to_mf_location_to_hashtags_ranked_by_model.iteritems():
###        for location, hashtags_ranked_by_model in mf_location_to_hashtags_ranked_by_model.iteritems():
###<<<<<<< HEAD
###            for hashtag, score in hashtags_ranked_by_model:
###                if location not in mf_to_location_to_mf_hashtag_to_mf_model_id_to_score:
###                    mf_to_location_to_mf_hashtag_to_mf_model_id_to_score[location] = defaultdict(dict)
###                mf_to_location_to_mf_hashtag_to_mf_model_id_to_score[location][location][model_id] = score
####                print model_id, location, hashtag, score
###
###    print 'x'
###        
####    for location, ltuo_hashtag_and_perct in mf_location_to_ideal_hashtags_rank.iteritems():
####        for hashtag, perct in ltuo_hashtag_and_perct:
####            print location, hashtag, perct
###=======
####            print model_id, location, hashtags_ranked_by_model
###            for hashtag, score in hashtags_ranked_by_model:
###                mf_hashtag_to_mf_model_id_to_score[hashtag][model_id] = score
###    
###    for location, ltuo_hashtag_and_perct in mf_location_to_ideal_hashtags_rank.iteritems():
###        for hashtag, perct in ltuo_hashtag_and_perct:
###            if hashtag in mf_hashtag_to_mf_model_id_to_score:
###                mf_hashtag_to_mf_model_id_to_score[hashtag]['value_to_predict'] = perct
###                yield location, mf_hashtag_to_mf_model_id_to_score[hashtag]
###
###for data in FileIO.iterateJsonFromFile('data/linear_regression'):
###    for fv in get_feature_vectors(data):
###        print fv
###>>>>>>> branch 'master' of ssh://git@github.com/kykamath/hashtags_and_geo.git
####            mf_model_id_to_score = {}
####            for model_id, mf_location_to_hashtags_ranked_by_model in \
####                    mf_model_id_to_mf_location_to_hashtags_ranked_by_model.iteritems():
###                
###                
###                
###                
