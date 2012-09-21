'''
Created on Sep 19, 2012

@author: kykamath
'''
from collections import defaultdict
from itertools import chain, groupby
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

NUM_OF_HASHTAGS = 100

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
                yield location, hashtag, perct, mf_hashtag_to_mf_model_id_to_score[hashtag]

def split_feature_vectors_into_test_and_training(feature_vectors):
    time_units = map(itemgetter('tu'), feature_vectors)
    time_units = sorted(list(set(time_units)))
#    feature_vectors.sort(key=itemgetter('tu'))
    test_time_unit = time_units[int(len(time_units)*(1-TESTING_RATIO))]
    train_feature_vectors, test_feature_vectors = [], []
    for feature_vector in feature_vectors:
        if feature_vector['tu'] > test_time_unit: test_feature_vectors.append(feature_vector)
        else: train_feature_vectors.append(feature_vector)
    return (train_feature_vectors, test_feature_vectors)

class EvaluationMetric(object):
    @staticmethod
    def accuracy(best_hashtags, predicted_hashtags, num_of_hashtags):
        return len(set(best_hashtags).intersection(set(predicted_hashtags)))/float(num_of_hashtags)
    @staticmethod
    def impact(best_hashtags, predicted_hashtags, hashtags_dist):
        total_perct_for_best_hashtags = sum([hashtags_dist.get(h, 0.0) for h in best_hashtags])
        total_perct_for_predicted_hashtags = sum([hashtags_dist.get(h, 0.0) for h in predicted_hashtags])
        return total_perct_for_predicted_hashtags/float(total_perct_for_best_hashtags)
    
class LearningToRank(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(LearningToRank, self).__init__(*args, **kwargs)
        self.mf_location_to_feature_vectors = defaultdict(list)
    def map_data_to_feature_vectors(self, key, line):
        if False: yield # I'm a generator!
        data = cjson.decode(line)
        for location, hashtag, actual_score, feature_vector in get_feature_vectors(data):
            self.mf_location_to_feature_vectors[location].append({
                                                                  'tu': data['tu'],
                                                                  'hashtag': hashtag,
                                                                  'actual_score': actual_score,
                                                                  'feature_vector': feature_vector
                                                                  })
    def map_final_data_to_feature_vectors(self):
        for location, feature_vectors in self.mf_location_to_feature_vectors.iteritems():
            yield location, feature_vectors
    def _get_parameter_names_to_values(self, train_feature_vectors):
        column_names = ['value_to_predict'] + LIST_OF_MODELS 
        mf_column_name_to_column_data = defaultdict(list)
        train_feature_vectors = map(itemgetter('feature_vector'), train_feature_vectors)
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
        return R_Helper.get_parameter_values(model)
    
    def red_feature_vectors_to_model(self, location, lo_feature_vector):
        column_names = ['value_to_predict'] + LIST_OF_MODELS 
        mf_column_name_to_column_data = defaultdict(list)
        feature_vectors = list(chain(*lo_feature_vector))
        train_feature_vectors, test_feature_vectors = split_feature_vectors_into_test_and_training(feature_vectors)
        if train_feature_vectors and test_feature_vectors:
            mf_parameter_names_to_values = dict(self._get_parameter_names_to_values(train_feature_vectors))
            lo_ltuo_hashtag_and_actual_score_and_feature_vector =\
                                    zip(
                                        [(tu, map(
                                                      itemgetter('hashtag', 'actual_score', 'feature_vector'),
                                                      it_feature_vectors)
                                                  )
                                            for tu, it_feature_vectors in 
                                                groupby(test_feature_vectors, key=itemgetter('tu'))
                                            ]
                                       )
            for ltuo_hashtag_and_actual_score_and_feature_vector in \
                    lo_ltuo_hashtag_and_actual_score_and_feature_vector:
                yield location, ltuo_hashtag_and_actual_score_and_feature_vector
#            for ltuo_hashtag_and_actual_score_and_feature_vector in \
#                    lo_ltuo_hashtag_and_actual_score_and_feature_vector:
#                ltuo_hashtag_and_actual_score_and_predicted_score =\
#                            map(lambda (hashtag, actual_score, feature_vector): 
#                                    (
#                                     hashtag,
#                                     actual_score,
#                                     R_Helper.get_predicted_value(mf_parameter_names_to_values, feature_vector)
#                                    ),
#                                ltuo_hashtag_and_actual_score_and_feature_vector)
#                yield location, ltuo_hashtag_and_actual_score_and_predicted_score
            
#            for ltuo_hashtag_and_actual_score_and_feature_vector in\
#                     lo_ltuo_hashtag_and_actual_score_and_feature_vector:
#                ltuo_hashtag_and_actual_score_and_score =\
#                                    map(lambda (hashtag, actual_score, feature_vector): 
#                                            (
#                                             hashtag,
#                                             actual_score,
#                                             R_Helper.get_predicted_value(mf_parameter_names_to_values, feature_vector)
#                                            ),
#                                        ltuo_hashtag_and_actual_score_and_feature_vector)
#                ltuo_hastag_and_actual_score = map(itemgetter(0, 1), ltuo_hashtag_and_actual_score_and_score)
#                ltuo_hastag_and_score = map(itemgetter(0, 2), ltuo_hashtag_and_actual_score_and_score)
#                ltuo_hastag_and_actual_score.sort(key=itemgetter(1))
#                ltuo_hastag_and_score.sort(key=itemgetter(1))
#                yield location, ltuo_hastag_and_actual_score
#                yield location, ltuo_hastag_and_score
##                print 'x'
            
#            for test_feature_vectors in lo_test_feature_vectors:
                
                
#            yield location, [len(train_feature_vectors), len(test_feature_vectors), mf_parameter_names_to_values]
    def steps(self):
        return [self.mr(
                    mapper=self.map_data_to_feature_vectors,
                    mapper_final=self.map_final_data_to_feature_vectors,
                    reducer=self.red_feature_vectors_to_model)
                ]

if __name__ == '__main__':
    LearningToRank.run()