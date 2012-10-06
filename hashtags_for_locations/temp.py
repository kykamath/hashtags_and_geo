#import simplejson as json
#import lxml
#from lxml import objectify

#
#s = '<row Id="114" PostTypeId="2" ParentId="102" CreationDate="2010-09-13T19:59:19.280" Score="1" ViewCount="" Body="&lt;p&gt;I rarely use the moto-droid hard keyboard. It boils down to your behavior eventually. For long emails I found myself using swype to draft it and then the soft keyboard to revise and send messages. I use the soft keyboard for anything needing less than 10 seconds of time to compose/complete.&lt;/p&gt;&#xA;&#xA;&lt;p&gt;The hardware keyboard has only been useful for the arrow keys since the droid doesnt have a trackball and touching the screen rarely gets the cursor where you want it. My next phone will not likely have a hard keyboard. I thought i would use it for gaming, but none of the devs are making games that really use the hard keys.&lt;/p&gt;&#xA;&#xA;&lt;p&gt;If you are simply looking for a peripheral keyboard you may want to see if changing up your input type and typing behavior saves you the problem.&lt;/p&gt;&#xA;" OwnerUserId="80" LastActivityDate="2010-09-13T19:59:19.280" />'
#
#s1 = 'sds'
#s1.replace('\n', '')
#replaprint S.replace('SPAM', 'EGGS')   ce_all
#print s1
##print s[:4]=='<row'
#exit()

import numpy as np
gap_perct = 0.02
print np.arange(gap_perct,1+gap_perct,gap_perct)

#from library.file_io import FileIO
#import sys
#
#def convert_row_to_dict(row):
#    try:
#        row = row.replace('\n', '')
#        row = row.replace('\r', '')
#        data = {}
#        for c in row[5:-2].split('" '):
#            if c:
#                c+='"'
#                print c
#                key, value = c.split('="')
#                value=value[:-1]
#                data[key] = value
#        if data: return data
#    except Exception as e:
#        print 'Caught expection', e
#        pass
#    
#def convert_file_to_json(input_file, output_file):
#    for line in FileIO.iterateLinesFromFile(input_file):
#        if line[:4]=='<row': 
#            data = convert_row_to_dict(line)
#            if data: FileIO.writeToFileAsJson(data, output_file)
#
#if __name__ == '__main__':
#    input_file = sys.argv[1]
#    output_file = sys.argv[2]
#    convert_file_to_json(input_file, output_file)
    
#print convert_row_to_dict(s)

#obj = objectify.fromstring("<Book><price>1.50</price><author>W. Shakespeare</author></Book>")  
#obj = objectify.fromstring(s)   
#
#class objectJSONEncoder(json.JSONEncoder):
#    """A specialized JSON encoder that can handle simple lxml objectify types
#       >>> from lxml import objectify
#       >>> obj = objectify.fromstring("<Book><price>1.50</price><author>W. Shakespeare</author></Book>")       
#       >>> objectJSONEncoder().encode(obj)
#       '{"price": 1.5, "author": "W. Shakespeare"}'       
#    """
#    def default(self,o):
#       if isinstance(o, lxml.objectify.IntElement):
#           return int(o)
#       if isinstance(o, lxml.objectify.NumberElement) or isinstance(o, lxml.objectify.FloatElement):
#           return float(o)
#       if isinstance(o, lxml.objectify.ObjectifiedDataElement):
#           return str(o)
#       if hasattr(o, '__dict__'):
#           #For objects with a __dict__, return the encoding of the __dict__
#           return o.__dict__
#       return json.JSONEncoder.default(self, o)
#   
#print objectJSONEncoder().encode(obj)

#def ps1(A):
#    unique_elements = set(A)
#    for index, item in enumerate(A):
#        if item in unique_elements:
#            unique_elements.remove(item)
#        if not unique_elements: return index
#        
#def ps2(A):
#    unique_elements = set()
#    possible_elements = set(range(len(A)))
#    for index, item in enumerate(A):
#        if item in possible_elements: 
#            possible_elements.remove(item)
#            unique_elements.add(item)
#        if not possible_elements: return index
#    for index, item in enumerate(A):
#        if item in unique_elements:
#            unique_elements.remove(item)
#        if not unique_elements: return index
#
##A = range(5)
##A[0] = 2;  A[1] = 2;  A[2] = 1
##A[3] = 0;  A[4] = 1
##print ps(A)
#
#from itertools import combinations    
#def number_of_disc_intersections ( A ):
#    num_of_disc_intersections = 0
#    for (pos1, rad1), (pos2, rad2) in combinations(enumerate(A), 2):
#        if abs(pos1-pos2)<=(rad1+rad2): num_of_disc_intersections+=1
#        if num_of_disc_intersections>10000000: return -1
#    return num_of_disc_intersections
#        
##A = [1, 5, 2, 1, 4, 0] 
##print number_of_disc_intersections(A)
#
#def stone_wall(H):
#    def block_can_be_used(block_len, target_wall_len_left):
#        if block_len <= target_wall_len_left: return True
#    blocks_used = 0
#    current_block_lens = []
#    for current_wall_len in H:
#        current_block_lens_used = []
#        current_wall_len_left = current_wall_len
#        for current_block_len in current_block_lens:
#            if block_can_be_used(current_block_len, current_wall_len_left):
#                current_wall_len_left-=current_block_len
#                current_block_lens_used.append(current_block_len)
#            else: break
#        if current_wall_len_left>0:
#            blocks_used+=1
#            current_block_lens_used.append(current_wall_len_left)
#        current_block_lens = current_block_lens_used
#    return blocks_used
#
##H = range(9)
##H[0] = 8;    H[1] = 8;    H[2] = 5       
##H[3] = 7;    H[4] = 9;    H[5] = 8       
##H[6] = 7;    H[7] = 4;    H[8] = 8
##print stone_wall(H)
#
#
#
##def equi ( A ):
##    left_sum = 0
##    right_sum = sum(A)
##    for index, item in enumerate(A):
##        left_sum+=A[index]
##        right_sum-=A[index]
##        if left_sum==right_sum: return index
##    return -1
##
##A = [-7, 1, 5, 2, -4, 3, 0]
##print equi(A)
#from pprint import pprint
#
#def countries_count ( A ):
#    country_id = []
#    num_of_countries = 0
#    M = len(A)
#    N = 0
#    for item in A:
#        N = len(item)
#        country_id.append([-1 for i in range(len(item))])
#    for i, items in enumerate(A):
#        for j, color in enumerate(items):
#            current_country_id = country_id[i][j]
#            if current_country_id==-1:
#                num_of_countries+=1 
#                current_country_id = num_of_countries
#                country_id[i][j] = current_country_id
#            valid_directions = filter(
#                                          lambda (i_,j_): i_<M and j_<N and i_>=0 and j_>=0,
#                                          [(i+1,j), (i-1,j), (i, j-1), (i,j+1)]
#                                      )
#            ltuo_neighbor_color_and_direction = [(A[i_][j_], (i_,j_)) for i_,j_ in valid_directions]
#            ltuo_neighbor_color_and_direction = filter(
#                                                       lambda (nc, d): nc==color,
#                                                       ltuo_neighbor_color_and_direction
#                                                       )
#            for _, (i_,j_) in ltuo_neighbor_color_and_direction:
#                country_id[i_][j_] = current_country_id
#    country_set = set()
#    for items in country_id:
#        for item in items: country_set.add(item)
#    return len(country_set)
##            print i, j, ltuo_neighbor_color_and_direction
#
#
#A = [[5, 4, 4], [4, 3, 4], [3, 2, 4], [2, 2, 2], [3, 3, 4], [1, 4, 4], [4, 1, 1]]
#
##print countries_count(A)
#
#def symmetryPoint ( S ):
#    for i in range(len(S)):
#        if S[:i] == S[i+1:][::-1]: return i
#    return -1
#
#def symmetryPoint1 ( S ):
#    reversed_S = S[::-1] 
#    N = len(S)
#    for i in range(N):
#        if S[:i] == reversed_S[:N-i-1]: return i
#    return -1
#
#S = 'racecar'
#print symmetryPoint(S)
#
###import matplotlib.pyplot as plt
###import numpy as np
###from scipy.stats import gaussian_kde
####data = [1.5]*7 + [2.5]*2 + [3.5]*8 + [4.5]*3 + [5.5]*1 + [6.5]*8
###data = [1.5]*7 + [5]*2 + [3.5]*8 + [7.5]*3
###density = gaussian_kde(data)
###xs = np.linspace(0,8,200)
###density.covariance_factor = lambda : .25
###density._compute_covariance()
###plt.plot(xs,density(xs), c='y')
###plt.fill_between(xs,density(xs),0,color='r')
####plt.hist(data)
###plt.show()
##
###from itertools import combinations
###
###print list(combinations(range(20),2))
##
##import random
##import numpy as np
###
###observed_data = [-1,1,1,-1,-1,-1,-1,-1,-1]
###
###iteration_data = [random.sample([-1,1],1)[0] for i in range(len(observed_data))]
###print actual_data
###print observed_data, np.mean()
##
###samples = [[54, 51, 58, 44, 55, 52, 42, 47, 58, 46], [54, 73, 53, 70, 73, 68, 52, 65, 65]]
##class MonteCarloSimulation(object):
##    '''
##    Part of this code was got from the implementation in the book "Statistics is Easy!" By Dennis Shasha and
##    Manda Wilson
##    '''
##    NUM_OF_SIMULATIONS = 10000
##    @staticmethod
##    def _shuffle(grps):
##        num_grps = len(grps)
##        pool = []
##        # pool all values
##        for i in range(num_grps):
##            pool.extend(grps[i])
##        # mix them up
##        random.shuffle(pool)
##        # reassign to groups of same size as original groups
##        new_grps = []
##        start_index = 0
##        end_index = 0
##        for i in range(num_grps):
##            end_index = start_index + len(grps[i])
##            new_grps.append(pool[start_index:end_index])
##            start_index = end_index
##        return new_grps
##    @staticmethod
##    # subtracts group a mean from group b mean and returns result
##    def _meandiff(grpA, grpB):
##        return sum(grpB) / float(len(grpB)) - sum(grpA) / float(len(grpA))
##
##    @staticmethod
##    def probability_of_data_extracted_from_same_sample(sample1, sample2):
##        ''' Difference between Two Means Significance Test
##        '''
##        samples = [sample1, sample2] 
##        a, b = 0, 1
##        observed_mean_diff = MonteCarloSimulation._meandiff(samples[a], samples[b])
##        count = 0
##        num_shuffles = MonteCarloSimulation.NUM_OF_SIMULATIONS
##        for i in range(num_shuffles):
##            new_samples = MonteCarloSimulation._shuffle(samples)
##            mean_diff = MonteCarloSimulation._meandiff(new_samples[a], new_samples[b])
##            # if the observed difference is negative, look for differences that are smaller
##            # if the observed difference is positive, look for differences that are greater
##            if observed_mean_diff < 0 and mean_diff <= observed_mean_diff: count = count + 1
##            elif observed_mean_diff >= 0 and mean_diff >= observed_mean_diff: count = count + 1
##        return (count / float(num_shuffles))
##
##observed_data = [54, 51, 58, 44, 55, 52, 42, 47, 58, 46]
##expected_data = [54, 73, 53, 70, 73, 68, 52, 65, 65]
##
###observed_data = [0,0,0,0,0,0,0,0,0]
#####observed_data = [0,1,1,0,1,0,1,0,1]
#####observed_data = [random.sample([0,1],1)[0] for i in range(len(observed_data))]
###expected_data = [random.sample([0,1],1)[0] for i in range(len(observed_data))]
##
##print observed_data, expected_data
##print MonteCarloSimulation.probability_of_data_extracted_from_same_sample(observed_data, expected_data)
##
#########################################
####
#### Output
####
#########################################
###
###print "Observed difference of two means: %.2f" % observed_mean_diff 
###print count, "out of", num_shuffles, "experiments had a difference of two means",
###if observed_mean_diff < 0:
###    print "less than or equal to",
###else:
###    print "greater than or equal to",
###print "%.2f" % observed_mean_diff, "."
###print "The chance of getting a difference of two means",
###if observed_mean_diff < 0:
###    print "less than or equal to",
###else:
###    print "greater than or equal to",
###print "%.2f" % observed_mean_diff, "is", (count / float(num_shuffles)), "."
##
####!/usr/bin/env python
###"""
###See pcolor_demo2 for a much faster way of generating pcolor plots
###"""
###from __future__ import division
###from pylab import *
###
###def func3(x,y):
###    return (1- x/2 + x**5 + y**3)*exp(-x**2-y**2)
###
###
#### make these smaller to increase the resolution
###dx, dy = 0.05, 0.05
###
###x = arange(-3.0, 3.0, dx)
###y = arange(-3.0, 3.0, dy)
###X,Y = meshgrid(x, y)
###
####Z = func3(X, Y)
###
###Z = [[1,2,3],[1,2,3],[1,3,3]]
###
###
###ax = subplot(111)
###im = imshow(Z, cmap=cm.jet)
####im.set_interpolation('nearest')
####im.set_interpolation('bicubic')
###im.set_interpolation('bilinear')
####ax.set_image_extent(-3, 3, -3, 3)
###
###show()
###gap_perct = 0.1
###occ_times_at_gap_perct = range(10)
###ltuo_perct_and_occ_time = [(int((gap_perct*i+gap_perct)*100), j)for i, j in enumerate(occ_times_at_gap_perct)]
###for perct1, occ_time1 in ltuo_perct_and_occ_time:
###    for perct2, occ_time2 in ltuo_perct_and_occ_time:
###        print perct1, perct2, max(occ_time2-occ_time1, 0.0)
###for i, val in enumerate(l):
###    for 
##
###import numpy as np
###from operator import itemgetter
###
###def get_items_at_gap(input_list, gap_perct):
###    return map(
###                  lambda index: input_list[index],
###                  map(lambda index: int(index)-1, np.arange(gap_perct,1+gap_perct,gap_perct)*100)
###              )
###
###print get_items_at_gap(range(100), 0.2)
###print map(lambda index: int(index)-1, np.arange(0.02,1+0.02,0.02)*100)
###print input_list[1]
##
###print map(itemgetter(map(int, np.arange(0,1,0.02)*100)), input_list)
##
###for perct in xrange()
###list_len = len(input_list)
###print input_list[int(perct*list_len)]
##
##
###def m(l):
###    l[0] = -1
###    
###def m2(l):
###    l = [-1 for i in range(10)]
###
###l = list([i for i in range(10)])
###a=[1,2,3]
###b=list(a)
###print a, b
###a[0]=-1
###print a, b
####print l
####m(l)
####print l
####m2(l)
####print l
####from library.file_io import FileIO
####
####for data in FileIO.iterateJsonFromFile('linear_regression_small'):
#####    print data.leys()
####    for loc in data['mf_location_to_ideal_hashtags_rank']:
####        print data['tu'], loc, sum(zip(*data['mf_location_to_ideal_hashtags_rank'][loc])[1])
####    exit()
####    
#####import numpy as np
#####from operator import itemgetter
######def get_occurrences_stats(occurrences1, occurrences2):
######        no_of_occurrences_after_appearing_in_location, no_of_occurrences_before_appearing_in_location = 0., 0.
######        occurrences1=sorted(occurrences1)
######        occurrences2=sorted(occurrences2)
######        no_of_total_occurrences_between_location_pair = len(occurrences1)*len(occurrences2)*1.
######        for occurrence1 in occurrences1:
######            for occurrence2 in occurrences2:
######                if occurrence1<occurrence2: no_of_occurrences_after_appearing_in_location+=1
######                elif occurrence1>occurrence2: no_of_occurrences_before_appearing_in_location+=1
######        return no_of_occurrences_after_appearing_in_location, no_of_occurrences_before_appearing_in_location, no_of_total_occurrences_between_location_pair
######
######def get_influence_scores(location_occurrences, neighbor_location_occurrences):
######        (no_of_occurrences_after_appearing_in_location, \
######         no_of_occurrences_before_appearing_in_location, \
######         no_of_total_occurrences_between_location_pair) =\
######            get_occurrences_stats(location_occurrences, neighbor_location_occurrences)
######        total_nof_occurrences = float(len(location_occurrences) + len(neighbor_location_occurrences))
######        ratio_of_occurrences_in_location = len(location_occurrences)/total_nof_occurrences
######        ratio_of_occurrences_in_neighbor_location = len(neighbor_location_occurrences)/total_nof_occurrences
######        return ( ratio_of_occurrences_in_location*no_of_occurrences_after_appearing_in_location - ratio_of_occurrences_in_neighbor_location*no_of_occurrences_before_appearing_in_location) / no_of_total_occurrences_between_location_pair
######    
######for location_object in FileIO.iterateJsonFromFile('data/40.6000_-73.9500'):
######    tuples_of_neighbor_location_and_pure_influence_score = []
######    location_hashtag_set = set(location_object['hashtags'])
######    for neighbor_location, map_from_hashtag_to_tuples_of_occurrences_and_time_range in location_object['links'].iteritems():
######        if neighbor_location in ['50.7500_2.1750', '35.5250_-5.8000', '27.5500_-16.6750']:
######            pure_influence_scores = []
#######            ht_n_score = []
######            for hashtag, (neighbor_location_occurrences, time_range) in map_from_hashtag_to_tuples_of_occurrences_and_time_range.iteritems():
######                if hashtag in location_object['hashtags']:
######                    location_occurrences = location_object['hashtags'][hashtag][0]
######                    pure_influence_scores.append(get_influence_scores(location_occurrences, neighbor_location_occurrences))
#######                    ht_n_score.append((hashtag, get_influence_scores(location_occurrences, neighbor_location_occurrences)))
######            neighbor_location_hashtag_set = set(map_from_hashtag_to_tuples_of_occurrences_and_time_range.keys())
######            for hashtag in location_hashtag_set.difference(neighbor_location_hashtag_set): pure_influence_scores.append(1.0)
######            for hashtag in neighbor_location_hashtag_set.difference(location_hashtag_set): pure_influence_scores.append(-1.0)
######            mean_pure_influence_score = np.mean(pure_influence_scores)
######            tuples_of_neighbor_location_and_pure_influence_score.append([neighbor_location, mean_pure_influence_score])
######        tuples_of_neighbor_location_and_pure_influence_score= sorted(tuples_of_neighbor_location_and_pure_influence_score, key=itemgetter(1))
######    print tuples_of_neighbor_location_and_pure_influence_score
#####
#####
#####from collections import defaultdict
#####
#####def get_feature_vectors(data):
#####    mf_model_id_to_mf_location_to_hashtags_ranked_by_model =\
#####                                                         data['mf_model_id_to_mf_location_to_hashtags_ranked_by_model']
#####    mf_location_to_ideal_hashtags_rank = data['mf_location_to_ideal_hashtags_rank']
#####    conf = data['conf']
#####    tu = data['tu']
#####    
#####    mf_to_location_to_mf_hashtag_to_mf_model_id_to_score = {}
#####    for model_id, mf_location_to_hashtags_ranked_by_model in \
#####            mf_model_id_to_mf_location_to_hashtags_ranked_by_model.iteritems():
#####        for location, hashtags_ranked_by_model in mf_location_to_hashtags_ranked_by_model.iteritems():
#####<<<<<<< HEAD
#####            for hashtag, score in hashtags_ranked_by_model:
#####                if location not in mf_to_location_to_mf_hashtag_to_mf_model_id_to_score:
#####                    mf_to_location_to_mf_hashtag_to_mf_model_id_to_score[location] = defaultdict(dict)
#####                mf_to_location_to_mf_hashtag_to_mf_model_id_to_score[location][location][model_id] = score
######                print model_id, location, hashtag, score
#####
#####    print 'x'
#####        
######    for location, ltuo_hashtag_and_perct in mf_location_to_ideal_hashtags_rank.iteritems():
######        for hashtag, perct in ltuo_hashtag_and_perct:
######            print location, hashtag, perct
#####=======
######            print model_id, location, hashtags_ranked_by_model
#####            for hashtag, score in hashtags_ranked_by_model:
#####                mf_hashtag_to_mf_model_id_to_score[hashtag][model_id] = score
#####    
#####    for location, ltuo_hashtag_and_perct in mf_location_to_ideal_hashtags_rank.iteritems():
#####        for hashtag, perct in ltuo_hashtag_and_perct:
#####            if hashtag in mf_hashtag_to_mf_model_id_to_score:
#####                mf_hashtag_to_mf_model_id_to_score[hashtag]['value_to_predict'] = perct
#####                yield location, mf_hashtag_to_mf_model_id_to_score[hashtag]
#####
#####for data in FileIO.iterateJsonFromFile('data/linear_regression'):
#####    for fv in get_feature_vectors(data):
#####        print fv
#####>>>>>>> branch 'master' of ssh://git@github.com/kykamath/hashtags_and_geo.git
######            mf_model_id_to_score = {}
######            for model_id, mf_location_to_hashtags_ranked_by_model in \
######                    mf_model_id_to_mf_location_to_hashtags_ranked_by_model.iteritems():
#####                
#####                
#####                
#####                
