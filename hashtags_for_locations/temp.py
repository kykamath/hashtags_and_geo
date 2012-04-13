from hashtags_for_locations.plots import GeneralAnalysis
from library.geo import getLocationFromLid, isWithinBoundingBox
from hashtags_for_locations.settings import US_BOUNDARY
tuples_of_location_and_tuples_of_neighbor_location_and_transmission_score = GeneralAnalysis.load_tuples_of_location_and_tuples_of_neighbor_location_and_transmission_score()
for location, tuples_of_neighbor_location_and_transmission_score in tuples_of_location_and_tuples_of_neighbor_location_and_transmission_score:
    if isWithinBoundingBox(getLocationFromLid(location.replace('_', ' ')), US_BOUNDARY):
        tuples_of_outgoing_location_and_transmission_score = filter(lambda (neighbor_location, transmission_score): transmission_score>0, tuples_of_neighbor_location_and_transmission_score)
        print location, len(tuples_of_outgoing_location_and_transmission_score), int(len(tuples_of_outgoing_location_and_transmission_score)*0.25)

#from pylab import *
#import numpy as np
#from matplotlib.transforms import Bbox
#from matplotlib.path import Path
#from matplotlib.patches import Rectangle
#
#rect = Rectangle((-1, -1), 2, 2, facecolor="#aaaaaa")
#gca().add_patch(rect)
#bbox = Bbox.from_bounds(-1, -1, 2, 2)
#
#for i in range(12):
#    vertices = (np.random.random((4, 2)) - 0.5) * 6.0
#    vertices = np.ma.masked_array(vertices, [[False, False], [True, True], [False, False], [False, False]])
#    path = Path(vertices)
#    if path.intersects_bbox(bbox):
#        color = 'r'
#    else:
#        color = 'b'
#    plot(vertices[:,0], vertices[:,1], color=color)
#
#show()
#
#
##from library.geo import getLocationFromLid, plotPointsOnWorldMap
##import matplotlib.pyplot as plt
##import matplotlib
##from settings import data_analysis_folder
##from library.classes import GeneralMethods
##from library.file_io import FileIO
##
##SOURCE_COLOR = 'r'
##OUTGOING_COLOR = 'g'
##INCOMING_COLOR = 'm'
##
##
##
##def plot_locations_on_map(source_location_lid, tuples_of_outgoing_location_and_transmission_score, tuples_of_incoming_location_and_transmission_score):
##    source_location = getLocationFromLid(source_location_lid.replace('_', ' '))
##    output_file = data_analysis_folder%GeneralMethods.get_method_id()+'%s.png'%source_location_lid
##    print output_file
##    def plot_locations(tuples_of_location_and_transmission_score):
##        locations, transmission_scores = zip(*sorted(
##                                               tuples_of_location_and_transmission_score,
##                                               key=lambda (location, transmission_score): abs(transmission_score)
##                                               ))
##        locations = [getLocationFromLid(location.replace('_', ' ')) for location in locations]
##        transmission_scores = [abs(transmission_score) for transmission_score in transmission_scores]
##        sc = plotPointsOnWorldMap(locations, blueMarble=False, bkcolor='#CFCFCF', c=transmission_scores, cmap=matplotlib.cm.winter,  lw = 0)
##        plt.colorbar(sc)
##        plotPointsOnWorldMap([source_location], blueMarble=False, bkcolor='#CFCFCF', c=SOURCE_COLOR, lw = 0)
##    
###    outgoing_locations = [getLocationFromLid(location.replace('_', ' ')) for location in outgoing_locations]
###    incoming_locations = [getLocationFromLid(location.replace('_', ' ')) for location in incoming_locations]
##    plt.subplot(211)
##    plot_locations(tuples_of_outgoing_location_and_transmission_score)
##    plt.title('Influences')
###    plotPointsOnWorldMap(outgoing_locations, blueMarble=False, bkcolor='#CFCFCF', c=OUTGOING_COLOR, lw = 0)
###    plotPointsOnWorldMap([source_location], blueMarble=False, bkcolor='#CFCFCF', c=SOURCE_COLOR, lw = 0)
##    plt.subplot(212)
##    plot_locations(tuples_of_incoming_location_and_transmission_score)
##    plt.title('Gets influenced by')
###    plotPointsOnWorldMap(outgoing_locations, blueMarble=False, bkcolor='#CFCFCF', c=INCOMING_COLOR, lw = 0)
###    plotPointsOnWorldMap([source_location], blueMarble=False, bkcolor='#CFCFCF', c=SOURCE_COLOR, lw = 0)
###    plt.show()
##    FileIO.createDirectoryForFile(output_file)
##    plt.savefig(output_file)
##    plt.clf()
##    
##source_location = '-12.3250_-37.7000'
###outgoing_locations = ['-0.7250_-47.8500', '-29.0000_-50.7500', '-21.7500_-42.7750', '-18.8500_-47.8500', '-10.8750_-36.9750', '-7.2500_-35.5250', '-2.9000_-41.3250', '-15.2250_-47.8500', '-22.4750_-43.5000', '-21.7500_-48.5750']
###incoming_locations = ['-3.6250_-38.4250', '-7.9750_-34.8000', '-15.9500_-48.5750', '-5.0750_-34.8000', '-9.4250_-47.8500', '-19.5750_-43.5000', '-2.1750_-44.2250', '-21.0250_-50.0250', '-29.7250_-50.7500', '-23.2000_-47.1250']
##
##tuples_of_outgoing_location_and_transmission_score = [['-20.3000_-53.6500', 9.2710695389300001e-05], ['-2.1750_-59.4500', 0.00046732702929899999], ['-5.0750_-42.7750', 0.0013858182161799999], ['-21.7500_-42.0500', 0.0036017416225700002], ['-15.2250_-47.1250', 0.0039704092844000003], ['-21.0250_-42.7750', 0.0053146258503399997], ['-22.4750_-46.4000', 0.0053747079639900001], ['-23.2000_-51.4750', 0.0058884980760000004], ['-27.5500_-50.7500', 0.00690519797663], ['-22.4750_-45.6750', 0.0072545748836500001], ['-23.9250_-46.4000', 0.0079496229731300001], ['-21.7500_-48.5750', 0.0081234582549599994], ['-22.4750_-43.5000', 0.0084389697236899996], ['-15.2250_-47.8500', 0.010085978836000001], ['-2.9000_-41.3250', 0.010843554593600001], ['-7.2500_-35.5250', 0.011064006387300001], ['-10.8750_-36.9750', 0.0127504300626], ['-18.8500_-47.8500', 0.0139438961612], ['-21.7500_-42.7750', 0.0149801587302], ['-29.0000_-50.7500', 0.015006558178800001], ['-0.7250_-47.8500', 0.017551628744699999]]
##tuples_of_incoming_location_and_transmission_score = [['-3.6250_-38.4250', -0.0208049886621], ['-7.9750_-34.8000', -0.014448197425700001], ['-15.9500_-48.5750', -0.0141851381607], ['-5.0750_-34.8000', -0.013941138662899999], ['-9.4250_-47.8500', -0.013909238909200001], ['-19.5750_-43.5000', -0.012846055309099999], ['-2.1750_-44.2250', -0.0127789279963], ['-21.0250_-50.0250', -0.0119277043619], ['-29.7250_-50.7500', -0.011310067278399999], ['-23.2000_-47.1250', -0.010535163139300001], ['-6.5250_-34.8000', -0.0088291357177500002], ['-23.2000_-46.4000', -0.0084152462097600005], ['-27.5500_-47.8500', -0.0080329101337499997], ['-23.2000_-45.6750', -0.00801652422882], ['-31.9000_-52.2000', -0.0075195923650199998], ['-25.3750_-48.5750', -0.0073247213893600002], ['-12.3250_-38.4250', -0.00654708021192], ['-26.8250_-48.5750', -0.00648750492463], ['-9.4250_-35.5250', -0.0062693755173799997], ['-22.4750_-44.9500', -0.0059606481481499999], ['-26.1000_-48.5750', -0.0053065165460100001], ['-20.3000_-39.8750', -0.0046276941657399998], ['-5.0750_-36.9750', -0.0033138928972300001], ['-5.8000_-34.8000', -0.0030856092436999999], ['-23.2000_-50.7500', -0.0030580048147700002], ['-20.3000_-49.3000', -0.0027396214896200002], ['-22.4750_-42.7750', -0.0027021326667199998], ['-22.4750_-47.1250', -0.00197884536838], ['-19.5750_-39.8750', -0.00130811227625], ['-21.0250_-47.1250', -3.3138255302099997e-05]]
##
##
##plot_locations_on_map(source_location, tuples_of_outgoing_location_and_transmission_score, tuples_of_incoming_location_and_transmission_score)
##
###def get_no_of_after_and_before_occurrences(occurrences1, occurrences2):
###    no_of_after_occurrences, no_of_before_occurrences = 0., 0.
###    occurrences1=sorted(occurrences1)
###    occurrences2=sorted(occurrences2)
###    no_of_total_occurrences = len(occurrences1)*len(occurrences2)*1.
###    for occurrence1 in occurrences1:
###        for occurrence2 in occurrences2:
###            if occurrence1<occurrence2: no_of_after_occurrences+=1
###            elif occurrence1>occurrence2: no_of_before_occurrences+=1
###    return no_of_after_occurrences, no_of_before_occurrences, no_of_total_occurrences
###def get_occurrence_direction_score(no_of_after_occurrences, no_of_before_occurrences, no_of_total_occurrences):
###    return (no_of_after_occurrences - no_of_before_occurrences) / no_of_total_occurrences
###def get_weighted_occurrence_direction_score(neighboring_location_weight, hashtag_weight, no_of_after_occurrences, no_of_before_occurrences, no_of_total_occurrences):
###    return neighboring_location_weight*hashtag_weight*get_occurrence_direction_score(no_of_after_occurrences, no_of_before_occurrences, no_of_total_occurrences)
###
###
###input1 = [1,2,3,4]
###input2 = [3,4,5,6]
###
###no_of_after_occurrences, no_of_before_occurrences, no_of_total_occurrences = get_no_of_after_and_before_occurrences(input1, input2)
###print no_of_after_occurrences, no_of_before_occurrences, no_of_total_occurrences
###print get_occurrence_direction_score(no_of_after_occurrences, no_of_before_occurrences, no_of_total_occurrences)
###print get_weighted_occurrence_direction_score(0.5, 0.5, no_of_after_occurrences, no_of_before_occurrences, no_of_total_occurrences)
###
####def count_non_flips((reduced_no_of_non_flips, reduced_previously_selected_model), (current_ep_time_unit, current_selected_model)):
####    if reduced_previously_selected_model==current_selected_model: reduced_no_of_non_flips+=1.0 
####    return (reduced_no_of_non_flips, current_selected_model)
####
####tuples_of_ep_time_unit_and_selected_model = zip([1,1,2,3,3,4], [1,1,1,1,1])
####print reduce(count_non_flips, tuples_of_ep_time_unit_and_selected_model, (0.0, None))
####
#####from numpy import arange,sqrt, random, linalg
#####from multiprocessing import Pool
#####
#####global counter
#####counter = 0
#####def cb(r):
#####    global counter
#####    print counter, r
#####    counter +=1
#####
#####def det((i,j)):
#####    return i+j
#####
#####class T:
#####    @staticmethod
#####    def runExperiment():
#####        po = Pool()
#####        #for i in xrange(1,300):
#####        #    j = random.normal(1,1,(100,100))
#####        #    po.apply_async(det,(j,),callback=cb)
#####            
#####        po.map_async(det, ((i,j) for i,j in zip(xrange(1,10), xrange(1,10))), callback=cb)
#####        po.close()
#####        po.join()
#####        print counter
#####
#####
#####T.runExperiment()
