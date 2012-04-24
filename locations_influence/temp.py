def get_misrank_accuracy((real_location_rank, locations_order_for_hashtag)):
    position = locations_order_for_hashtag.index(real_location_rank)
    def count_greater_than(current_count, (real_location_rank, predicted_location_rank)):
        if real_location_rank < predicted_location_rank: current_count+=1
        return current_count
    def count_lesser_than(current_count, (real_location_rank, predicted_location_rank)):
        if real_location_rank > predicted_location_rank: current_count+=1
        return current_count
    left_side_location_ranks = locations_order_for_hashtag[:position]
    right_side_location_ranks = locations_order_for_hashtag[position+1:]
    total_misranked_locations = reduce(count_greater_than, zip([real_location_rank]*len(left_side_location_ranks), left_side_location_ranks), 0.0) \
                                    + reduce(count_lesser_than, zip([real_location_rank]*len(right_side_location_ranks), right_side_location_ranks), 0.0)
    return total_misranked_locations/(len(locations_order_for_hashtag)-1)

#locations_order_for_hashtag = [3,2,1,4,5]
#locations_order_for_hashtag = [1,2,3,4,5]
locations_order_for_hashtag = [5,4,3,2,1]
#locations_order_for_hashtag = [5, 1,2,3,4]
real_location_rank = 2


rank_accuracies = map(
                      get_misrank_accuracy,
                      zip(locations_order_for_hashtag, [locations_order_for_hashtag]*len(locations_order_for_hashtag))
                      )
print rank_accuracies
