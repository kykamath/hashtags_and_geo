def count_non_flips((reduced_no_of_non_flips, reduced_previously_selected_model), (current_ep_time_unit, current_selected_model)):
    if reduced_previously_selected_model==current_selected_model: reduced_no_of_non_flips+=1.0 
    return (reduced_no_of_non_flips, current_selected_model)

tuples_of_ep_time_unit_and_selected_model = zip([1,1,2,3,3,4], [1,1,1,1,1])
print reduce(count_non_flips, tuples_of_ep_time_unit_and_selected_model, (0.0, None))

#from numpy import arange,sqrt, random, linalg
#from multiprocessing import Pool
#
#global counter
#counter = 0
#def cb(r):
#    global counter
#    print counter, r
#    counter +=1
#
#def det((i,j)):
#    return i+j
#
#class T:
#    @staticmethod
#    def runExperiment():
#        po = Pool()
#        #for i in xrange(1,300):
#        #    j = random.normal(1,1,(100,100))
#        #    po.apply_async(det,(j,),callback=cb)
#            
#        po.map_async(det, ((i,j) for i,j in zip(xrange(1,10), xrange(1,10))), callback=cb)
#        po.close()
#        po.join()
#        print counter
#
#
#T.runExperiment()
