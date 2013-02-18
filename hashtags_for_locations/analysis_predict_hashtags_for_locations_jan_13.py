'''
Created on Jan 22, 2013

@author: krishnakamath
'''
from datetime import datetime
from datetime import timedelta
from library.file_io import FileIO
from library.geo import getLocationFromLid
from library.geo import isWithinBoundingBox
from library.mrjobwrapper import runMRJob
from mr_predict_hashtags_for_locations_jan_13 import HashtagsByModelsByLocations
from mr_predict_hashtags_for_locations_jan_13 import ModelPerformance
from pprint import pprint
import cjson

TIME_UNIT_IN_SECONDS = 60 * 60
US_BOUNDARY = [[24.527135, -127.792969], [49.61071, -59.765625]]

dfs_data_folder = 'hdfs:///user/kykamath/geo/hashtags/2011-09-01_2011-11-01/'
dfs_input = dfs_data_folder + '360_120/100/linear_regression'

analysis_folder = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/predict_hashtags_for_locations/%s'
f_hashtags_by_models_by_locations = analysis_folder % 'hashtags_by_models_by_locations'
f_model_performance = analysis_folder % 'model_performance'

class MRAnalysis(object):
    @staticmethod
    def get_input_files():
        for j in range(1, 7):
            predictionTimeInterval = timedelta(seconds=j * TIME_UNIT_IN_SECONDS)
            yield '%s/%s_%s/100/linear_regression' % (dfs_data_folder, 360, predictionTimeInterval.seconds / 60)
    @staticmethod
    def hashtags_by_models_by_locations():
        runMRJob(
                    HashtagsByModelsByLocations,
                    f_hashtags_by_models_by_locations,
#                     [dfs_input],
                    MRAnalysis.get_input_files(),
                    jobconf={'mapred.reduce.tasks':500, 'mapred.task.timeout': 86400000}
                 )
    @staticmethod
    def model_performance():
        runMRJob(
                    ModelPerformance,
                    f_model_performance,
                    [dfs_input],
                    jobconf={'mapred.reduce.tasks':500, 'mapred.task.timeout': 86400000}
                 )
    @staticmethod
    def run():
        MRAnalysis.hashtags_by_models_by_locations()
#        MRAnalysis.model_performance()

class ModelAnalysis(object):
    @staticmethod
    def hashtags_by_models_by_locations():
        '''
        [1316570400.0,"40.7450_-73.9500",{
        "sharing_probability" : ["dudesthatsaynohomo","terriblenamesforavagina","cgi2011","foino20desetembro","sub12anos"],
        "transmitting_probability" : ["cgi2011","dudesthatsaynohomo","terriblenamesforavagina","foino20desetembro","takewallstreet"],
        "coverage_distance" : ["epatcon","dudesthatsaynohomo","cgi2011","takewallstreet","terriblenamesforavagina"],
        "coverage_probability" : ["teamenzomusic","takewallstreet","cgi2011","miscellaney","epatcon"],
        "random" : ["cgi2011"],
        "ideal_ranking" : [["faze3",0.285714285714],["terriblenamesforavagina",0.285714285714],["cgi2011",0.142857142857],["dudesthatsaynohomo",0.142857142857],["takewallstreet",0.142857142857]],
        "greedy" : ["cgi2011"]
        },
        [["sharing_probability",3],["transmitting_probability",4],["coverage_distance",4],["coverage_probability",2],["random",1],["greedy",1]]]
        '''
        data = '[["faze3",0.285714285714],["terriblenamesforavagina",0.285714285714],["cgi2011",0.142857142857],' + \
                                        '["dudesthatsaynohomo",0.142857142857],["takewallstreet",0.142857142857]]'
        mf_hashtag_to_value = dict(cjson.decode(data))
        def accuracy(hashtags, mf_hashtag_to_value):
            return len(set(hashtags).intersection(set(mf_hashtag_to_value.keys())))/5.
        def impact(hashtags, mf_hashtag_to_value): 
            total = sum(mf_hashtag_to_value.values())
            val = sum([mf_hashtag_to_value.get(h, 0.0) for h in hashtags])
            return val/total
        def get_valid_location((location, mf_model_id_to_hashtags)):
            location = getLocationFromLid(location.replace('_', ' '))
            return isWithinBoundingBox(location, US_BOUNDARY)
        def get_distance(ltuo_meta_and_ltuo_model_id_and_similarity_score):
            _, _, _, ltuo_model_id_and_similarity_score = ltuo_meta_and_ltuo_model_id_and_similarity_score
            mf_model_id_to_similarity_score = dict(ltuo_model_id_and_similarity_score)
            return mf_model_id_to_similarity_score['transmitting_probability'] - \
                                                                            mf_model_id_to_similarity_score['greedy']
        ltuo_meta_and_ltuo_model_id_and_similarity_score = []
        for data in FileIO.iterateJsonFromFile(f_hashtags_by_models_by_locations, True):
            ltuo_location_and_mf_model_id_to_hashtags = filter(get_valid_location, data['locations'].iteritems())
            for location, mf_model_id_to_hashtags in ltuo_location_and_mf_model_id_to_hashtags:
                ltuo_model_id_and_similarity_score = []
                ideal_ranking = set(zip(*mf_model_id_to_hashtags['ideal_ranking'])[0])
                for model_id, hashtags in mf_model_id_to_hashtags.iteritems():
                    if model_id not in 'ideal_ranking':
                        hashtags = list(set(zip(*hashtags)[0]))
                        ltuo_model_id_and_similarity_score += \
                                                        [(model_id, len(ideal_ranking.intersection(set(hashtags))))]
                ltuo_meta_and_ltuo_model_id_and_similarity_score += \
                                [[data['tu'], location, mf_model_id_to_hashtags, ltuo_model_id_and_similarity_score]]
        ltuo_meta_and_ltuo_model_id_and_similarity_score.sort(key=get_distance)
        for data in ltuo_meta_and_ltuo_model_id_and_similarity_score:
            if data[0] == 1316570400 and data[1] == '40.7450_-73.9500':
                print data[:2]
                mf_model_to_hashtags = data[2]
                for model, hashtags in mf_model_to_hashtags.iteritems():
                    if type(hashtags[0]) != type([]):
                        print '%0.2f'%accuracy(hashtags, mf_hashtag_to_value), \
                                                        '%0.2f'%impact(hashtags, mf_hashtag_to_value), model, hashtags
                exit()
#            print data[1], data[-1]
    @staticmethod
    def run():
        ModelAnalysis.hashtags_by_models_by_locations()
if __name__ == '__main__':
#    MRAnalysis.run()
    ModelAnalysis.run()    
