"""
Make a "broken" horizontal bar plot, ie one with gaps
"""
import matplotlib.pyplot as plt

fig = plt.figure()
#ax = fig.add_subplot(111)
plt.broken_barh([ (110, 30), (150, 10) ] , (10, 9), facecolors='blue')
plt.broken_barh([ (10, 50), (100, 20),  (130, 10)] , (20, 9),
                facecolors=('red', 'yellow', 'green'), alpha=0.5, lw=0)
#ax.set_ylim(5,35)
#ax.set_xlim(0,200)
#ax.set_xlabel('seconds since start')
#ax.set_yticks([15,25])
#ax.set_yticklabels(['Bill', 'Jim'])
#ax.grid(True)
#ax.annotate('race interrupted', (61, 25),
#            xytext=(0.8, 0.9), textcoords='axes fraction',
#            arrowprops=dict(facecolor='black', shrink=0.05),
#            fontsize=16,
#            horizontalalignment='right', verticalalignment='top')

plt.show()

#import numpy as np
#from sklearn.cluster.affinity_propagation_ import AffinityPropagation
#
#def cluster_locations_based_on_influence_scores(ltuo_locations_and_influence_scores):
#    def similarity_matrix(similarity_matrix, (current_point, all_points)):
#        similarity_matrix.append([1./(np.abs(current_point - point)+1)for point in all_points])
#        return similarity_matrix
#    locations, influence_scores = zip(*ltuo_locations_and_influence_scores)
#    S = np.array(reduce(
#                    similarity_matrix,
#                    zip(influence_scores, [influence_scores]*len(influence_scores)),
#                    []
#                ))
#    af = AffinityPropagation().fit(S)
#    return (len(af.cluster_centers_indices_), zip(locations, af.labels_))
#
#ltuo_locations_and_influence_scores = [
#                                       ('a',1),
#                                       ('b',2),
#                                       ('c',10),
#                                       ('d',12),
#                                       ('e',11),
#                                       ]
#print cluster_locations_based_on_influence_scores(ltuo_locations_and_influence_scores)
#
#
##def get_misrank_accuracy((real_location_rank, locations_order_for_hashtag)):
##    position = locations_order_for_hashtag.index(real_location_rank)
##    def count_greater_than(current_count, (real_location_rank, predicted_location_rank)):
##        if real_location_rank < predicted_location_rank: current_count+=1
##        return current_count
##    def count_lesser_than(current_count, (real_location_rank, predicted_location_rank)):
##        if real_location_rank > predicted_location_rank: current_count+=1
##        return current_count
##    left_side_location_ranks = locations_order_for_hashtag[:position]
##    right_side_location_ranks = locations_order_for_hashtag[position+1:]
##    total_misranked_locations = reduce(count_greater_than, zip([real_location_rank]*len(left_side_location_ranks), left_side_location_ranks), 0.0) \
##                                    + reduce(count_lesser_than, zip([real_location_rank]*len(right_side_location_ranks), right_side_location_ranks), 0.0)
##    return total_misranked_locations/(len(locations_order_for_hashtag)-1)
##
##locations_order_for_hashtag = [3,2,1,4,5]
##locations_order_for_hashtag = [1,2,3,4,5]
##locations_order_for_hashtag = [5,4,3,2,1]
###locations_order_for_hashtag = [5, 1,2,3,4]
##real_location_rank = 2
##
##
##rank_accuracies = map(
##                      get_misrank_accuracy,
##                      zip(locations_order_for_hashtag, [locations_order_for_hashtag]*len(locations_order_for_hashtag))
##                      )
##print rank_accuracies
