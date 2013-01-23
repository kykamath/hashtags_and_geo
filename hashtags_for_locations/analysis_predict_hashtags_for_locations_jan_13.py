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
from pprint import pprint
import cjson

TIME_UNIT_IN_SECONDS = 60 * 60
US_BOUNDARY = [[24.527135, -127.792969], [49.61071, -59.765625]]

dfs_data_folder = 'hdfs:///user/kykamath/geo/hashtags/2011-09-01_2011-11-01/'
dfs_input = dfs_data_folder + '360_120/100/linear_regression'

analysis_folder = '/mnt/chevron/kykamath/data/geo/hashtags/hashtags_for_locations/predict_hashtags_for_locations/%s'
f_hashtags_by_models_by_locations = analysis_folder % 'hashtags_by_models_by_locations'

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
    def run():
        MRAnalysis.hashtags_by_models_by_locations()

def temp():
    def get_valid_location((location, mf_model_id_to_hashtags)):
        location = getLocationFromLid(location.replace('_', ' '))
        return isWithinBoundingBox(location, US_BOUNDARY)
    def get_distance(ltuo_meta_and_ltuo_model_id_and_similarity_score):
        _, _, _, ltuo_model_id_and_similarity_score = ltuo_meta_and_ltuo_model_id_and_similarity_score
        mf_model_id_to_similarity_score = dict(ltuo_model_id_and_similarity_score)
        return mf_model_id_to_similarity_score['sharing_probability'] - mf_model_id_to_similarity_score['greedy']
    ltuo_meta_and_ltuo_model_id_and_similarity_score = []
    for data in FileIO.iterateJsonFromFile(f_hashtags_by_models_by_locations, True):
#        locations = data['locations']
#        if len(locations) > 10:
        ltuo_location_and_mf_model_id_to_hashtags = filter(get_valid_location, data['locations'].iteritems())
        for location, mf_model_id_to_hashtags in ltuo_location_and_mf_model_id_to_hashtags:
            ltuo_model_id_and_similarity_score = []
            ideal_ranking = set(zip(*mf_model_id_to_hashtags['ideal_ranking'])[0])
            for model_id, hashtags in mf_model_id_to_hashtags.iteritems():
                if model_id not in 'ideal_ranking':
                    ltuo_model_id_and_similarity_score += [(model_id, len(ideal_ranking.intersection(set(hashtags))))]
            ltuo_meta_and_ltuo_model_id_and_similarity_score += \
                                [[data['tu'], location, mf_model_id_to_hashtags, ltuo_model_id_and_similarity_score]]
#    for i in ltuo_meta_and_ltuo_model_id_and_similarity_score:
#        print i
#    exit()
    ltuo_meta_and_ltuo_model_id_and_similarity_score.sort(key=get_distance)
    for data in ltuo_meta_and_ltuo_model_id_and_similarity_score:
        print cjson.encode(data)
        print data[1], data[-1]
#    s = '''
#     {"tu": 1315810800.0, "locations": {"-23.6350_-46.5450": {"sharing_probability": ["infocartola20mil", "whatyouwantev", "50vezesdamiao", "mantegaogagao"], "transmitting_probability": ["infocartola20mil", "whatyouwantev", "50vezesdamiao", "mantegaogagao"], "coverage_distance": ["whatyouwantev", "mantegaogagao", "infocartola20mil", "megapatriota"], "coverage_probability": ["webclipesinais", "megapatriota", "manteg\u00e3o     gag\u00e3o", "brasildeouro"], "random": ["webclipesinais", "parabenspet", "mantegaogagao", "50vezesdamiao"], "ideal_ranking": [["promobieberm", 0.6], ["edin\u00e9iapopstar", 0.1], ["infocartola20mil", 0.1], ["mantegaogagao", 0.1]], "greedy": ["whatyouwantev", "megapatriota", "mantegaogagao", "infocartola20mil"]}, "19.4300_-99.0350": {"sharing_probability": ["10cancionesquenuncafaltanenmiipod", "mientalelamadreawereve     rtumorro", "estabamosbienhastaque", "millondewerevertumorro"], "transmitting_probability": ["10cancionesquenuncafaltanenmiipod", "mientalelamadreawerevertumorro", "estabamosbienhastaque", "millondewerevertumorro"], "coverage_distance": ["10cancionesquenuncafaltanenmiipod", "estabamosbienhastaque", "mientalelamadreawerevertumorro", "millondewerevertumorro"], "coverage_probability": ["2001\u5e749\u670811\u65e5\u4f55\u5     1e6\u3067\u4f55\u3057\u3066\u305f", "estabamosbienhastaque", "2\u79d2\u3067\u3070\u308c\u308b\u5618\u3092\u3064\u3044\u3066\u304f\u3060\u3055\u3044", "millondewerevertumorro"], "random": ["millondewerevertumorro", "2\u79d2\u3067\u3070\u308c\u308b\u5618\u3092\u3064\u3044\u3066\u304f\u3060\u3055\u3044", "mientalelamadreawerevertumorro", "10cancionesquenuncafaltanenmiipod"], "ideal_ranking": [["estabamosbienhastaque", 0     .5], ["yoamoaespinozapaz", 0.4], ["10cancionesquenuncafaltanenmiipod", 0.05], ["mientalelamadreawerevertumorro", 0.05]], "greedy": ["10cancionesquenuncafaltanenmiipod", "estabamosbienhastaque", "mientalelamadreawerevertumorro", "2001\u5e749\u670811\u65e5\u4f55\u51e6\u3067\u4f55\u3057\u3066\u305f"]}, "-14.2100_-53.0700": {"sharing_probability": ["10cancionesquenuncafaltanenmiipod", "infocartola20mil", "whatyouwantev",      "50vezesdamiao"], "transmitting_probability": ["10cancionesquenuncafaltanenmiipod", "infocartola20mil", "whatyouwantev", "50vezesdamiao"], "coverage_distance": ["10cancionesquenuncafaltanenmiipod", "esdemagallanico", "juegoschilenos", "frasesdepublicidades"], "coverage_probability": ["benjoo", "juegoschilenos", "saludypobreza", "esdemagallanico"], "random": ["frasesdepublicidades", "10cancionesquenuncafaltanenmiipod     ", "esdemagallanico", "juegoschilenos"], "ideal_ranking": [["esdemagallanico", 0.375], ["juegoschilenos", 0.25], ["10cancionesquenuncafaltanenmiipod", 0.125], ["bocapuntero", 0.125]], "greedy": ["10cancionesquenuncafaltanenmiipod", "esdemagallanico", "juegoschilenos", "frasesdepublicidades"]}}}
#    '''
#    data = cjson.decode(s)
#    for location, mf_model_id_to_hashtags in data['locations'].iteritems():
#        print location
#        for model_id, hashtags in mf_model_id_to_hashtags.iteritems():
#            if model_id == 'ideal_ranking':
#                print model_id
#                print sorted(zip(*hashtags)[0])
#            else:
#                print model_id
#                print sorted(hashtags)
if __name__ == '__main__':
    MRAnalysis.run()
#    temp()
    
