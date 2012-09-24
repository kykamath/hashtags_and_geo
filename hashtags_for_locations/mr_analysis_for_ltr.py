'''
Created on Sep 19, 2012

@author: kykamath
'''
from collections import defaultdict
from itertools import chain, groupby
from library.mrjobwrapper import ModifiedMRJob
from library.r_helper import R_Helper
from operator import itemgetter
import cjson
import numpy as np
import rpy2.robjects as robjects

R = robjects.r

LIST_OF_MODELS = [
                  'greedy',
                  'sharing_probability',
                  'transmitting_probability',
                  'coverage_distance',
                  'coverage_probability'
                  ]

TESTING_RATIO = 0.25

NUM_OF_HASHTAGS = 100

def get_feature_vectors(data):
    mf_model_id_to_mf_location_to_hashtags_ranked_by_model =\
                                                         data['mf_model_id_to_mf_location_to_hashtags_ranked_by_model']
    mf_location_to_ideal_hashtags_rank = data['mf_location_to_ideal_hashtags_rank']
    
    mf_location_to_mf_hashtag_to_mf_model_id_to_score = {}
    for model_id, mf_location_to_hashtags_ranked_by_model in \
            mf_model_id_to_mf_location_to_hashtags_ranked_by_model.iteritems():
        for location, hashtags_ranked_by_model in mf_location_to_hashtags_ranked_by_model.iteritems():
#            print model_id, location, hashtags_ranked_by_model
            for hashtag, score in hashtags_ranked_by_model:
                if location not in mf_location_to_mf_hashtag_to_mf_model_id_to_score:
                    mf_location_to_mf_hashtag_to_mf_model_id_to_score[location] = defaultdict(dict)
                mf_location_to_mf_hashtag_to_mf_model_id_to_score[location][hashtag][model_id] = score
    
    for location, mf_hashtag_to_mf_model_id_to_score in \
            mf_location_to_mf_hashtag_to_mf_model_id_to_score.iteritems():
        ltuo_hashtag_and_perct = []
        if location in mf_location_to_ideal_hashtags_rank: 
            ltuo_hashtag_and_perct = mf_location_to_ideal_hashtags_rank[location]
        mf_hashtag_to_value_to_predict = dict(ltuo_hashtag_and_perct)
        for hashtag, mf_model_id_to_score in mf_hashtag_to_mf_model_id_to_score.iteritems():
#            actual_score = None
#            if hashtag in mf_hashtag_to_value_to_predict:
#                actual_score = mf_hashtag_to_value_to_predict[hashtag]
            mf_model_id_to_score['value_to_predict'] = mf_hashtag_to_value_to_predict.get(hashtag, None)
            yield location, hashtag, mf_model_id_to_score
            
#    
#    
#    for location, ltuo_hashtag_and_perct in mf_location_to_ideal_hashtags_rank.iteritems():
#        for hashtag, perct in ltuo_hashtag_and_perct:
##            if hashtag in mf_hashtag_to_mf_model_id_to_score:
#            mf_location_to_mf_hashtag_to_mf_model_id_to_score[location][hashtag]['value_to_predict'] = perct
#            yield location, hashtag, perct, mf_location_to_mf_hashtag_to_mf_model_id_to_score[location][hashtag]

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
        return len(set(best_hashtags).intersection(set(predicted_hashtags)))/float(len(best_hashtags))
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
        for location, hashtag, feature_vector in get_feature_vectors(data):
            location_id = '%s++%s++%s'%(
                                            location,
                                            data['conf']['historyTimeInterval'],
                                            data['conf']['predictionTimeInterval']
                                        )
            self.mf_location_to_feature_vectors[location_id].append({
                                                                  'tu': data['tu'],
                                                                  'hashtag': hashtag,
                                                                  'actual_score': feature_vector['value_to_predict'],
                                                                  'feature_vector': feature_vector
                                                                  })
    def map_final_data_to_feature_vectors(self):
        for location, feature_vectors in self.mf_location_to_feature_vectors.iteritems():
            yield location, feature_vectors
    def _get_parameter_names_to_values(self, train_feature_vectors):
#        column_names = ['value_to_predict'] + LIST_OF_MODELS 
        mf_column_name_to_column_data = defaultdict(list)
        train_feature_vectors = map(itemgetter('feature_vector'), train_feature_vectors)
        for feature_vector in train_feature_vectors:
            if feature_vector['value_to_predict']:
                mf_column_name_to_column_data['value_to_predict'].append(feature_vector['value_to_predict'])
                for column_name in LIST_OF_MODELS:
                    mf_column_name_to_column_data[column_name].append(feature_vector.get(column_name, 0.0))
        data = {}
        for column_name, column_data in mf_column_name_to_column_data.iteritems():
            data[column_name] = robjects.FloatVector(column_data)
        if data:
            data_frame = robjects.DataFrame(data)
            prediction_variable = 'value_to_predict'
            predictor_variables = LIST_OF_MODELS
            model = R_Helper.linear_regression_model(
                                                     data_frame,
                                                     prediction_variable,
                                                     predictor_variables,
    #                                                 with_variable_selection=True
                                                    )
            return R_Helper.get_parameter_values(model)
    
    def red_feature_vectors_to_model(self, location, lo_feature_vector):
        feature_vectors = list(chain(*lo_feature_vector))
        train_feature_vectors, test_feature_vectors = split_feature_vectors_into_test_and_training(feature_vectors)
        filtered_train_feature_vectors = filter(lambda fv: len(fv['feature_vector'])>1, train_feature_vectors)
        filtered_test_feature_vectors = filter(lambda fv: len(fv['feature_vector'])>1, test_feature_vectors)
        
        if filtered_train_feature_vectors and filtered_test_feature_vectors:
            parameter_names_to_values = self._get_parameter_names_to_values(filtered_train_feature_vectors)
            if parameter_names_to_values:
                accuracy_mf_num_of_hashtags_to_metric_values = defaultdict(list)
                impact_mf_num_of_hashtags_to_metric_values = defaultdict(list)
                mf_parameter_names_to_values = dict(parameter_names_to_values)
                test_feature_vectors.sort(key=itemgetter('tu'))
                lo_ltuo_hashtag_and_actual_score_and_feature_vector =\
                                                    [(tu, map(
                                                              itemgetter('hashtag', 'actual_score', 'feature_vector'),
                                                              it_feature_vectors)
                                                          )
                                                        for tu, it_feature_vectors in 
                                                            groupby(test_feature_vectors, key=itemgetter('tu'))
                                                    ]
                    
                for tu, ltuo_hashtag_and_actual_score_and_feature_vector in \
                        lo_ltuo_hashtag_and_actual_score_and_feature_vector:
                    for _, __, fv in ltuo_hashtag_and_actual_score_and_feature_vector: del fv['value_to_predict']
                    ltuo_hashtag_and_actual_score_and_predicted_score =\
                                    map(lambda (hashtag, actual_score, feature_vector): 
                                            (
                                             hashtag,
                                             actual_score,
                                             R_Helper.get_predicted_value(mf_parameter_names_to_values, feature_vector)
                                            ),
                                        ltuo_hashtag_and_actual_score_and_feature_vector)
                    ltuo_hashtag_and_actual_score = [ (hashtag, actual_score)
                                                     for hashtag, actual_score, _ in
                                                            ltuo_hashtag_and_actual_score_and_predicted_score 
                                                        if actual_score!=None]
                    ltuo_hashtag_and_predicted_score = map(
                                                        itemgetter(0,2),
                                                        ltuo_hashtag_and_actual_score_and_predicted_score
                                                        )
                    
                    if ltuo_hashtag_and_actual_score and ltuo_hashtag_and_predicted_score:
                        
                        ltuo_hashtag_and_actual_score = sorted(
                                                               ltuo_hashtag_and_actual_score,
                                                               key=itemgetter(1),
                                                               reverse=True
                                                               )
                        ltuo_hashtag_and_predicted_score = sorted(
                                                               ltuo_hashtag_and_predicted_score,
                                                               key=itemgetter(1),
                                                               reverse=True
                                                               )
                        
                        for num_of_hashtags in range(1,25):
                            hashtags_dist = dict(ltuo_hashtag_and_actual_score)
                            actual_ordering_of_hashtags = zip(*ltuo_hashtag_and_actual_score)[0]
                            predicted_ordering_of_hashtags = zip(*ltuo_hashtag_and_predicted_score)[0]
                            
                            accuracy = EvaluationMetric.accuracy(
                                                                  actual_ordering_of_hashtags[:num_of_hashtags],
                                                                  predicted_ordering_of_hashtags[:num_of_hashtags],
                                                                  num_of_hashtags
                                                                )
                            impact = EvaluationMetric.impact(
                                                            actual_ordering_of_hashtags[:num_of_hashtags],
                                                            predicted_ordering_of_hashtags[:num_of_hashtags],
                                                            hashtags_dist
                                                          )
                            accuracy_mf_num_of_hashtags_to_metric_values[num_of_hashtags].append(accuracy)
                            impact_mf_num_of_hashtags_to_metric_values[num_of_hashtags].append(impact)
                yield location, accuracy_mf_num_of_hashtags_to_metric_values
                yield location, impact_mf_num_of_hashtags_to_metric_values
                                
#                data = location.split('++')
#                window_id = '%s_%s'%(data[1], data[2])
#                mf_metric_to_mf_num_of_hashtags_to_metric_value = defaultdict(dict)
#                for num_of_hashtags, metric_values in\
#                        accuracy_mf_num_of_hashtags_to_metric_values.iteritems():
#                    mf_metric_to_mf_num_of_hashtags_to_metric_value['accuracy'][num_of_hashtags] =\
#                                                                                                np.mean(metric_values)
#                for num_of_hashtags, metric_values in\
#                        impact_mf_num_of_hashtags_to_metric_values.iteritems():
#                    mf_metric_to_mf_num_of_hashtags_to_metric_value['impact'][num_of_hashtags] = np.mean(metric_values)
#                    
#                output_dict = {
#                               'window_id': window_id,
#                               'location': data[0],
#                               'mf_metric_to_mf_num_of_hashtags_to_metric_value': 
#                                                                        mf_metric_to_mf_num_of_hashtags_to_metric_value
#                            }
#                yield location, output_dict
                        
    def steps(self):
        return [self.mr(
                    mapper=self.map_data_to_feature_vectors,
                    mapper_final=self.map_final_data_to_feature_vectors,
                    reducer=self.red_feature_vectors_to_model
                )
            ]

if __name__ == '__main__':
    LearningToRank.run()