def get_no_of_after_and_before_occurrences(occurrences1, occurrences2):
    no_of_after_occurrences, no_of_before_occurrences = 0., 0.
    occurrences1=sorted(occurrences1)
    occurrences2=sorted(occurrences2)
    no_of_total_occurrences = len(occurrences1)*len(occurrences2)*1.
    for occurrence1 in occurrences1:
        for occurrence2 in occurrences2:
            if occurrence1<occurrence2: no_of_after_occurrences+=1
            elif occurrence1>occurrence2: no_of_before_occurrences+=1
    return no_of_after_occurrences, no_of_before_occurrences, no_of_total_occurrences
def get_occurrence_direction_score(no_of_after_occurrences, no_of_before_occurrences, no_of_total_occurrences):
    return (no_of_after_occurrences - no_of_before_occurrences) / no_of_total_occurrences
def get_weighted_occurrence_direction_score(neighboring_location_weight, hashtag_weight, no_of_after_occurrences, no_of_before_occurrences, no_of_total_occurrences):
    return neighboring_location_weight*hashtag_weight*get_occurrence_direction_score(no_of_after_occurrences, no_of_before_occurrences, no_of_total_occurrences)


input1 = [1,2,3,4]
input2 = [3,4,5,6]

no_of_after_occurrences, no_of_before_occurrences, no_of_total_occurrences = get_no_of_after_and_before_occurrences(input1, input2)
print no_of_after_occurrences, no_of_before_occurrences, no_of_total_occurrences
print get_occurrence_direction_score(no_of_after_occurrences, no_of_before_occurrences, no_of_total_occurrences)
print get_weighted_occurrence_direction_score(0.5, 0.5, no_of_after_occurrences, no_of_before_occurrences, no_of_total_occurrences)

#def count_non_flips((reduced_no_of_non_flips, reduced_previously_selected_model), (current_ep_time_unit, current_selected_model)):
#    if reduced_previously_selected_model==current_selected_model: reduced_no_of_non_flips+=1.0 
#    return (reduced_no_of_non_flips, current_selected_model)
#
#tuples_of_ep_time_unit_and_selected_model = zip([1,1,2,3,3,4], [1,1,1,1,1])
#print reduce(count_non_flips, tuples_of_ep_time_unit_and_selected_model, (0.0, None))
#
##from numpy import arange,sqrt, random, linalg
##from multiprocessing import Pool
##
##global counter
##counter = 0
##def cb(r):
##    global counter
##    print counter, r
##    counter +=1
##
##def det((i,j)):
##    return i+j
##
##class T:
##    @staticmethod
##    def runExperiment():
##        po = Pool()
##        #for i in xrange(1,300):
##        #    j = random.normal(1,1,(100,100))
##        #    po.apply_async(det,(j,),callback=cb)
##            
##        po.map_async(det, ((i,j) for i,j in zip(xrange(1,10), xrange(1,10))), callback=cb)
##        po.close()
##        po.join()
##        print counter
##
##
##T.runExperiment()
