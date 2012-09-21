'''
Created on Sep 19, 2012

@author: kykamath
'''
from collections import defaultdict
from itertools import chain
from library.mrjobwrapper import ModifiedMRJob
from library.r_helper import R_Helper
from operator import itemgetter
import rpy2.robjects as robjects
import cjson

R = robjects.r

LIST_OF_MODELS = [
                  'greedy',
                  'sharing_probability',
                  'transmitting_probability',
                  'coverage_distance',
                  'coverage_probability'
                  ]

TESTING_RATIO = 0.1

def get_feature_vectors(data):
    mf_model_id_to_mf_location_to_hashtags_ranked_by_model =\
                                                         data['mf_model_id_to_mf_location_to_hashtags_ranked_by_model']
    mf_location_to_ideal_hashtags_rank = data['mf_location_to_ideal_hashtags_rank']
    
    mf_hashtag_to_mf_model_id_to_score = defaultdict(dict)
    for model_id, mf_location_to_hashtags_ranked_by_model in \
            mf_model_id_to_mf_location_to_hashtags_ranked_by_model.iteritems():
        for location, hashtags_ranked_by_model in mf_location_to_hashtags_ranked_by_model.iteritems():
#            print model_id, location, hashtags_ranked_by_model
            for hashtag, score in hashtags_ranked_by_model:
                mf_hashtag_to_mf_model_id_to_score[hashtag][model_id] = score
    
    for location, ltuo_hashtag_and_perct in mf_location_to_ideal_hashtags_rank.iteritems():
        for hashtag, perct in ltuo_hashtag_and_perct:
            if hashtag in mf_hashtag_to_mf_model_id_to_score:
                mf_hashtag_to_mf_model_id_to_score[hashtag]['value_to_predict'] = perct
                yield location, mf_hashtag_to_mf_model_id_to_score[hashtag]

def split_feature_vectors_into_test_and_training(feature_vectors):
    feature_vectors.sort(key=itemgetter('tu'))
    feature_vectors = map(itemgetter('feature_vector'), feature_vectors)
    test_index = int(len(feature_vectors)*(1-TESTING_RATIO))
    return (feature_vectors[:test_index], feature_vectors[test_index:])

class EvaluationMetric(object):
    @staticmethod
    def accuracy(hashtags1, hashtags2, num_of_hashtags):
        hashtags1 = hashtags1[]
        return len(set(hashtags1).intersection(set(hashtags2)))/float(num_of_hashtags)
    @staticmethod
    def impact(hashtags1, hashtags2, hashtags_dist, num_of_hashtags):
#        return len(set(hashtags1).intersection(set(hashtags2))
    
class LearningToRank(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(LearningToRank, self).__init__(*args, **kwargs)
        self.mf_location_to_feature_vectors = defaultdict(list)
    def map_data_to_feature_vectors(self, key, line):
        if False: yield # I'm a generator!
        data = cjson.decode(line)
        for location, feature_vector in get_feature_vectors(data):
            self.mf_location_to_feature_vectors[location].append({'tu': data['tu'], 'feature_vector': feature_vector})
    def map_final_data_to_feature_vectors(self):
        for location, feature_vectors in self.mf_location_to_feature_vectors.iteritems():
            yield location, feature_vectors
    def red_feature_vectors_to_model(self, location, lo_feature_vector):
        column_names = ['value_to_predict'] + LIST_OF_MODELS 
        mf_column_name_to_column_data = defaultdict(list)
        feature_vectors = list(chain(*lo_feature_vector))
        train_feature_vectors, test_feature_vectors = split_feature_vectors_into_test_and_training(feature_vectors)
        if train_feature_vectors and test_feature_vectors:
            for feature_vector in train_feature_vectors:
                for column_name in column_names:
                    mf_column_name_to_column_data[column_name].append(feature_vector.get(column_name, 0.0))
            data = {}
            for column_name, column_data in mf_column_name_to_column_data.iteritems():
                data[column_name] = robjects.FloatVector(column_data)
            data_frame = robjects.DataFrame(data)
            prediction_variable = 'value_to_predict'
            predictor_variables = LIST_OF_MODELS
            model = R_Helper.linear_regression_model(
                                                     data_frame,
                                                     prediction_variable,
                                                     predictor_variables,
                                                     with_variable_selection=True
                                                    )
            parameter_names_and_values = R_Helper.get_parameter_values(model)
            yield location, [len(train_feature_vectors), len(test_feature_vectors), parameter_names_and_values]
    def steps(self):
        return [self.mr(
                    mapper=self.map_data_to_feature_vectors,
                    mapper_final=self.map_final_data_to_feature_vectors,
                    reducer=self.red_feature_vectors_to_model)
                ]

if __name__ == '__main__':
    LearningToRank.run()