'''
Created on Sep 19, 2012

@author: kykamath
'''
from collections import defaultdict
from itertools import chain, groupby
from library.classes import GeneralMethods
from library.mrjobwrapper import ModifiedMRJob
from library.r_helper import R_Helper
from operator import itemgetter
import cjson
import numpy as np
import random
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

NUM_OF_HASHTAGS = range(1, 25)

# Prediction method ids.
PREDICTION_METHOD_ID_FOLLOW_THE_LEADER = 'follow_the_leader'
PREDICTION_METHOD_ID_HEDGING = 'hedging'
PREDICTION_METHOD_ID_LEARNING_TO_RANK = 'learning_to_rank'

# Beta for hedging method
BETA = 0.5

def get_feature_vectors(data):
    mf_model_id_to_mf_location_to_hashtags_ranked_by_model =\
                                                         data['mf_model_id_to_mf_location_to_hashtags_ranked_by_model']
    mf_location_to_ideal_hashtags_rank = data['mf_location_to_ideal_hashtags_rank']
    
    mf_location_to_mf_hashtag_to_mf_model_id_to_score = {}
    for model_id, mf_location_to_hashtags_ranked_by_model in \
            mf_model_id_to_mf_location_to_hashtags_ranked_by_model.iteritems():
        for location, hashtags_ranked_by_model in mf_location_to_hashtags_ranked_by_model.iteritems():
            for hashtag, score in hashtags_ranked_by_model:
                if location not in mf_location_to_mf_hashtag_to_mf_model_id_to_score:
                    mf_location_to_mf_hashtag_to_mf_model_id_to_score[location] = defaultdict(dict)
                mf_location_to_mf_hashtag_to_mf_model_id_to_score[location][hashtag][model_id] = score
    
    for location, mf_hashtag_to_mf_model_id_to_score in \
            mf_location_to_mf_hashtag_to_mf_model_id_to_score.iteritems():
        ltuo_hashtag_and_perct = []
        yielded_hashtags = set()
        if location in mf_location_to_ideal_hashtags_rank: 
            ltuo_hashtag_and_perct = mf_location_to_ideal_hashtags_rank[location]
        mf_hashtag_to_value_to_predict = dict(ltuo_hashtag_and_perct)
        if mf_hashtag_to_value_to_predict:
            for hashtag, mf_model_id_to_score in mf_hashtag_to_mf_model_id_to_score.iteritems():
                mf_model_id_to_score['value_to_predict'] = mf_hashtag_to_value_to_predict.get(hashtag, None)
                yielded_hashtags.add(hashtag)
                yield location, hashtag, mf_model_id_to_score
            for hashtag, value_to_predict in mf_hashtag_to_value_to_predict.iteritems():
                if hashtag not in yielded_hashtags:
                    yielded_hashtags.add(hashtag)
                    yield location, hashtag, {'value_to_predict': value_to_predict}
            
def split_feature_vectors_into_test_and_training(feature_vectors):
    time_units = map(itemgetter('tu'), feature_vectors)
    time_units = sorted(list(set(time_units)))
    test_time_unit = time_units[int(len(time_units)*(1-TESTING_RATIO))]
    train_feature_vectors, test_feature_vectors = [], []
    for feature_vector in feature_vectors:
        if feature_vector['tu'] > test_time_unit: test_feature_vectors.append(feature_vector)
        else: train_feature_vectors.append(feature_vector)
    return (train_feature_vectors, test_feature_vectors)

class EvaluationMetric(object):
    ID_ACCURACY = 'accuracy'
    ID_IMPACT = 'impact'
    @staticmethod
    def accuracy(best_hashtags, predicted_hashtags, num_of_hashtags):
        return len(set(best_hashtags).intersection(set(predicted_hashtags)))/float(len(best_hashtags))
    @staticmethod
    def impact(best_hashtags, predicted_hashtags, hashtags_dist):
        total_perct_for_best_hashtags = sum([hashtags_dist.get(h, 0.0) for h in best_hashtags])
        total_perct_for_predicted_hashtags = sum([hashtags_dist.get(h, 0.0) for h in predicted_hashtags])
        return total_perct_for_predicted_hashtags/float(total_perct_for_best_hashtags)
    
class LearningToRank(object):
    @staticmethod
    def _get_parameter_names_to_values(train_feature_vectors):
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
    @staticmethod
    def get_performance_metrics(feature_vectors, *args, **kwargs):
        train_feature_vectors, test_feature_vectors = split_feature_vectors_into_test_and_training(feature_vectors)
        filtered_train_feature_vectors = filter(lambda fv: len(fv['feature_vector'])>1, train_feature_vectors)
        filtered_test_feature_vectors = filter(lambda fv: len(fv['feature_vector'])>1, test_feature_vectors)
        
        if filtered_train_feature_vectors and filtered_test_feature_vectors:
            parameter_names_to_values = LearningToRank._get_parameter_names_to_values(filtered_train_feature_vectors)
            if parameter_names_to_values:
                accuracy_mf_num_of_hashtags_to_metric_values = defaultdict(list)
                impact_mf_num_of_hashtags_to_metric_values = defaultdict(list)
                mf_parameter_names_to_values = dict(parameter_names_to_values)
                test_feature_vectors.sort(key=itemgetter('tu'))
                ltuo_tu_and_ltuo_hashtag_and_actual_score_and_feature_vector =\
                                                    [(tu, map(
                                                              itemgetter('hashtag', 'actual_score', 'feature_vector'),
                                                              it_feature_vectors)
                                                          )
                                                        for tu, it_feature_vectors in 
                                                            groupby(test_feature_vectors, key=itemgetter('tu'))
                                                    ]
                    
                for tu, ltuo_hashtag_and_actual_score_and_feature_vector in \
                        ltuo_tu_and_ltuo_hashtag_and_actual_score_and_feature_vector:
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
                        
                        for num_of_hashtags in NUM_OF_HASHTAGS:
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
                return (accuracy_mf_num_of_hashtags_to_metric_values, impact_mf_num_of_hashtags_to_metric_values)
        return {}, {}

class OnlineLearning():
    @staticmethod
    def get_performance_of_models_by_time_unit(feature_vectors):
            for fv in feature_vectors: 
                if 'value_to_predict' in fv['feature_vector']: del fv['feature_vector']['value_to_predict']
            feature_vectors.sort(key=itemgetter('tu'))
            ltuo_tu_and_ltuo_hashtag_and_actual_score_and_feature_vector =\
                                                [(tu, map(
                                                          itemgetter('hashtag', 'actual_score', 'feature_vector'),
                                                          it_feature_vectors)
                                                      )
                                                    for tu, it_feature_vectors in 
                                                        groupby(feature_vectors, key=itemgetter('tu'))
                                                ]
            ltuo_tu_and_ltuo_hashtag_and_actual_score_and_feature_vector.sort(key=itemgetter(0))
            mf_num_of_hashtags_to_ltuo_tu_and_mf_model_id_to_mf_metric_to_value = defaultdict(list)
            for tu, ltuo_hashtag_and_actual_score_and_feature_vector in \
                            ltuo_tu_and_ltuo_hashtag_and_actual_score_and_feature_vector:
                ltuo_observed_hastags_and_actual_score = [(hashtag, actual_score)
                                                          for hashtag, actual_score, _ in 
                                                                ltuo_hashtag_and_actual_score_and_feature_vector
                                                            if actual_score!=None
                                                        ]
                if ltuo_observed_hastags_and_actual_score:
                    ltuo_observed_hastags_and_actual_score.sort(key=itemgetter(1), reverse=True)
                    actual_ordering_of_hashtags = zip(*ltuo_observed_hastags_and_actual_score)[0]
                    hashtags_dist = dict(ltuo_observed_hastags_and_actual_score)
                    
                    mf_model_id_to_ltuo_hashtag_and_predicted_score = defaultdict(list)
                    mf_model_id_to_predicted_ordering_of_hashtags = {}
                    for hashtag, actual_score, fv in ltuo_hashtag_and_actual_score_and_feature_vector:
                        for model_id, predicted_score in fv.iteritems():
                            mf_model_id_to_ltuo_hashtag_and_predicted_score[model_id].append([hashtag, predicted_score])
                    for model_id, ltuo_hashtag_and_predicted_score in\
                            mf_model_id_to_ltuo_hashtag_and_predicted_score.items()[:]:
                        ltuo_hashtag_and_predicted_score.sort(key=itemgetter(1), reverse=True)
                        mf_model_id_to_predicted_ordering_of_hashtags[model_id] =\
                                                                            zip(*ltuo_hashtag_and_predicted_score)[0]
                    for num_of_hashtags in NUM_OF_HASHTAGS:
                        mf_model_id_to_mf_metric_to_value = {}
                        for model_id, predicted_ordering_of_hashtags in\
                                mf_model_id_to_predicted_ordering_of_hashtags.iteritems():
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
                            mf_model_id_to_mf_metric_to_value[model_id] =\
                                                                        dict([
                                                                            (EvaluationMetric.ID_ACCURACY, accuracy),
                                                                            (EvaluationMetric.ID_IMPACT, impact)
                                                                        ])
                        mf_num_of_hashtags_to_ltuo_tu_and_mf_model_id_to_mf_metric_to_value[num_of_hashtags]\
                            .append([tu, mf_model_id_to_mf_metric_to_value])
            return mf_num_of_hashtags_to_ltuo_tu_and_mf_model_id_to_mf_metric_to_value
    @staticmethod
    def get_performance_metrics(feature_vectors, get_best_model, update_losses_for_every_model):
        accuracy_mf_num_of_hashtags_to_metric_values = defaultdict(list)
        impact_mf_num_of_hashtags_to_metric_values = defaultdict(list)
        mf_num_of_hashtags_to_ltuo_tu_and_mf_model_id_to_mf_metric_to_value =\
                                                 OnlineLearning.get_performance_of_models_by_time_unit(feature_vectors)
        for num_of_hashtags, ltuo_tu_and_mf_model_id_to_mf_metric_to_value in \
                mf_num_of_hashtags_to_ltuo_tu_and_mf_model_id_to_mf_metric_to_value.iteritems():
#            accuracy_mf_model_id_to_cumulative_losses = dict([(model_id, 0.0) for model_id in LIST_OF_MODELS])
#            impact_mf_model_id_to_cumulative_losses = dict([(model_id, 0.0) for model_id in LIST_OF_MODELS])
            accuracy_mf_model_id_to_cumulative_losses = {}
            impact_mf_model_id_to_cumulative_losses = {}
            for tu, mf_model_id_to_mf_metric_to_value in ltuo_tu_and_mf_model_id_to_mf_metric_to_value:
                accuracy_mf_model_id_to_metric_value = {}
                impact_mf_model_id_to_metric_value = {}
                for model_id in LIST_OF_MODELS:
                    mf_metric_to_value = mf_model_id_to_mf_metric_to_value.get(model_id, {})
                    accuracy_mf_model_id_to_metric_value[model_id] =\
                                                            mf_metric_to_value.get(EvaluationMetric.ID_ACCURACY, 0.0)
                    impact_mf_model_id_to_metric_value[model_id] =\
                                                                mf_metric_to_value.get(EvaluationMetric.ID_IMPACT, 0.0)
                for mf_model_id_to_cumulative_losses, mf_model_id_to_metric_value, mf_num_of_hashtags_to_metric_values\
                        in \
                        [
                             (
                                  accuracy_mf_model_id_to_cumulative_losses,
                                  accuracy_mf_model_id_to_metric_value,
                                  accuracy_mf_num_of_hashtags_to_metric_values
                              ),
                             (
                                  impact_mf_model_id_to_cumulative_losses,
                                  impact_mf_model_id_to_metric_value,
                                  impact_mf_num_of_hashtags_to_metric_values
                              ),
                         ]:
                    best_model = get_best_model(mf_model_id_to_cumulative_losses)
                    mf_num_of_hashtags_to_metric_values[num_of_hashtags].append(mf_model_id_to_metric_value[best_model])
                    update_losses_for_every_model(
                                                      mf_model_id_to_metric_value,
                                                      best_model,
                                                      mf_model_id_to_cumulative_losses
                                                  )
        return accuracy_mf_num_of_hashtags_to_metric_values, impact_mf_num_of_hashtags_to_metric_values
    @staticmethod
    def follow_the_leader_get_best_model(mf_model_id_to_cumulative_losses):
        if not mf_model_id_to_cumulative_losses: return random.sample(LIST_OF_MODELS, 1)[0]
        else: 
            model_id_and_cumulative_loss = (None, ())
            for model_id in LIST_OF_MODELS: 
                if model_id in mf_model_id_to_cumulative_losses: 
                    model_id_and_cumulative_loss = min(
                                                        [
                                                             model_id_and_cumulative_loss, 
                                                             (model_id, mf_model_id_to_cumulative_losses[model_id])
                                                         ],
                                                        key=itemgetter(1)
                                                    )
            return model_id_and_cumulative_loss[0]
    @staticmethod
    def hedging_get_best_model(mf_model_id_to_cumulative_losses):
        if not mf_model_id_to_cumulative_losses: return random.sample(LIST_OF_MODELS, 1)[0]
        else:
            total_weight = float(sum(mf_model_id_to_cumulative_losses.values()))
            for model in mf_model_id_to_cumulative_losses.keys(): mf_model_id_to_cumulative_losses[model]/=total_weight 
            id_and_model_id_and_cumulative_losses = [(id, model_id, cumulative_loss)
                                                        for id, (model_id, cumulative_loss) in 
                                                            enumerate(mf_model_id_to_cumulative_losses.iteritems())
                                                    ]
            selected_id = GeneralMethods.weightedChoice(zip(*id_and_model_id_and_cumulative_losses)[2])
            return filter(lambda (id, model, _): id==selected_id, id_and_model_id_and_cumulative_losses)[0][1]
    @staticmethod
    def follow_the_leader_update_losses_for_every_model(
                                                        mf_model_id_to_metric_value,
                                                        best_model,
                                                        mf_model_id_to_cumulative_losses
                                                    ):
        for model_id, metric_value in mf_model_id_to_metric_value.iteritems():
            if model_id not in mf_model_id_to_cumulative_losses: mf_model_id_to_cumulative_losses[model_id] = 0.0
            mf_model_id_to_cumulative_losses[model_id]+=(1.0 - metric_value)
    @staticmethod
    def hedging_update_losses_for_every_model(
                                                mf_model_id_to_metric_value,
                                                best_model,
                                                mf_model_id_to_cumulative_losses
                                            ):
        for model_id, metric_value in mf_model_id_to_metric_value.iteritems():
            if model_id not in mf_model_id_to_cumulative_losses: mf_model_id_to_cumulative_losses[model_id] = 1.0
            mf_model_id_to_cumulative_losses[model_id]*=BETA**(1.0 - metric_value)

class PredictingHastagsForLocations(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(PredictingHastagsForLocations, self).__init__(*args, **kwargs)
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
    
    def _yield_results(
                           self,
                           prediction_method,
                           location,
                           accuracy_mf_num_of_hashtags_to_metric_values,
                           impact_mf_num_of_hashtags_to_metric_values
                       ):
        if accuracy_mf_num_of_hashtags_to_metric_values.items() and\
                impact_mf_num_of_hashtags_to_metric_values.items():
            data = location.split('++')
            window_id = '%s_%s'%(data[1], data[2])
            output_dict = {
                              'prediction_method': prediction_method,
                              'window_id': window_id,
                              'num_of_hashtags': -1,
                              'location': data[0],
                              'metric': '',
                              'metric_value': 0.0
                          }
            for num_of_hashtags, metric_values in\
                    accuracy_mf_num_of_hashtags_to_metric_values.iteritems():
                output_dict['metric'] = EvaluationMetric.ID_ACCURACY
                output_dict['num_of_hashtags'] = num_of_hashtags
                output_dict['metric_value'] = np.mean(metric_values)
                yield 'o_d', output_dict
            for num_of_hashtags, metric_values in\
                    impact_mf_num_of_hashtags_to_metric_values.iteritems():
                output_dict['metric'] = EvaluationMetric.ID_IMPACT
                output_dict['num_of_hashtags'] = num_of_hashtags
                output_dict['metric_value'] = np.mean(metric_values)
                yield 'o_d', output_dict
    
    def red_feature_vectors_to_measuring_unit_id_and_metric_value(self, location, lo_feature_vector):
        feature_vectors = list(chain(*lo_feature_vector))
        
        accuracy_mf_num_of_hashtags_to_metric_values, impact_mf_num_of_hashtags_to_metric_values=\
                                                                LearningToRank.get_performance_metrics(feature_vectors)
        for output in self._yield_results(
                                    PREDICTION_METHOD_ID_LEARNING_TO_RANK,
                                    location,
                                    accuracy_mf_num_of_hashtags_to_metric_values,
                                    impact_mf_num_of_hashtags_to_metric_values
                            ):
                yield output

        for prediction_method, get_best_model, update_losses_for_every_model in\
                [
                     (  
                        PREDICTION_METHOD_ID_FOLLOW_THE_LEADER,
                        OnlineLearning.follow_the_leader_get_best_model,
                        OnlineLearning.follow_the_leader_update_losses_for_every_model
                      ),
                     (
                        PREDICTION_METHOD_ID_HEDGING,
                        OnlineLearning.hedging_get_best_model,
                        OnlineLearning.hedging_update_losses_for_every_model
                      )
                 ]:
            accuracy_mf_num_of_hashtags_to_metric_values, impact_mf_num_of_hashtags_to_metric_values=\
                                                OnlineLearning.get_performance_metrics(
                                                                                        feature_vectors,
                                                                                        get_best_model,
                                                                                        update_losses_for_every_model
                                                                                    )
            for output in self._yield_results(
                                    prediction_method,
                                    location,
                                    accuracy_mf_num_of_hashtags_to_metric_values,
                                    impact_mf_num_of_hashtags_to_metric_values
                            ):
                yield output
    def jobs_to_evaluate_prediction_methods(self):
        return [self.mr(
                    mapper=self.map_data_to_feature_vectors,
                    mapper_final=self.map_final_data_to_feature_vectors,
                    reducer=self.red_feature_vectors_to_measuring_unit_id_and_metric_value
                )
            ] 
    
    def steps(self):
        return self.jobs_to_evaluate_prediction_methods()

class PerformanceOfPredictingMethodsByVaryingParameter(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(PerformanceOfPredictingMethodsByVaryingParameter, self).__init__(*args, **kwargs)
        self.mf_varying_parameter_to_metric_values = defaultdict(list)
    def map_data_to_num_of_hashtags_and_value(self, key, performance_data):
        if False: yield # I'm a generator!
        performance_data = cjson.decode(performance_data)
        num_of_hashtags = '%s::%s::%s::%s'%(
                                        performance_data['num_of_hashtags'],
                                        'num_of_hashtags',
                                        performance_data['metric'],
                                        performance_data['prediction_method']
                                    )
        self.mf_varying_parameter_to_metric_values[num_of_hashtags].append(performance_data['metric_value'])
    def map_data_to_prediction_time_interval_and_value(self, key, performance_data):
        if False: yield # I'm a generator!
        performance_data = cjson.decode(performance_data)
        historical_time_interval, prediction_time_interval = map(
                                                                 float,
                                                                 performance_data['window_id'].split('_')
                                                                 )
        if historical_time_interval==21600.0:
            prediction_time_interval = '%s::%s::%s::%s'%(
                                            prediction_time_interval,
                                            'prediction_time_interval',
                                            performance_data['metric'],
                                            performance_data['prediction_method']
                                        )
            self.mf_varying_parameter_to_metric_values[prediction_time_interval].append(
                                                                                    performance_data['metric_value']
                                                                                )
    def map_data_to_historical_time_interval_and_value(self, key, performance_data):
        if False: yield # I'm a generator!
        performance_data = cjson.decode(performance_data)
        historical_time_interval, prediction_time_interval = map(
                                                                 float,
                                                                 performance_data['window_id'].split('_')
                                                                 )
        if prediction_time_interval==3600.0:
            historical_time_interval = '%s::%s::%s::%s'%(
                                            historical_time_interval,
                                            'historical_time_interval',
                                            performance_data['metric'],
                                            performance_data['prediction_method']
                                        )
            self.mf_varying_parameter_to_metric_values[historical_time_interval].append(
                                                                                    performance_data['metric_value']
                                                                                )
    def map_final_data_to_varying_parameter_and_value(self):
        for varying_parameter, metric_values in self.mf_varying_parameter_to_metric_values.iteritems():
            yield varying_parameter, metric_values
    def red_varying_parameter_and_metric_values_to_performance_summary(self, varying_parameter_key, lo_metric_values):
        varying_parameter, varying_parameter_id, metric, prediction_method = varying_parameter_key.split('::')
        performance_summary = {
                               varying_parameter_id: float(varying_parameter),
                               'metric': metric,
                               'prediction_method': prediction_method,
                               'metric_value': np.mean(list(chain(*lo_metric_values)))
                               }
        yield 'o_d', performance_summary
    def jobs_for_performance_of_predicting_by_varying_num_of_hashtags(self):
        return [self.mr(
                        mapper=self.map_data_to_num_of_hashtags_and_value,
                        mapper_final=self.map_final_data_to_varying_parameter_and_value,
                        reducer=self.red_varying_parameter_and_metric_values_to_performance_summary
                    )
                ]
    def jobs_for_performance_of_predicting_by_varying_prediction_time_interval(self):
        return [self.mr(
                        mapper=self.map_data_to_prediction_time_interval_and_value,
                        mapper_final=self.map_final_data_to_varying_parameter_and_value,
                        reducer=self.red_varying_parameter_and_metric_values_to_performance_summary
                    )
                ]
    def jobs_for_performance_of_predicting_by_varying_historical_time_interval(self):
        return [self.mr(
                        mapper=self.map_data_to_historical_time_interval_and_value,
                        mapper_final=self.map_final_data_to_varying_parameter_and_value,
                        reducer=self.red_varying_parameter_and_metric_values_to_performance_summary
                    )
                ]
    def steps(self):
#        return self.jobs_for_performance_of_predicting_by_varying_num_of_hashtags()
#        return self.jobs_for_performance_of_predicting_by_varying_prediction_time_interval()
        return self.jobs_for_performance_of_predicting_by_varying_historical_time_interval()

if __name__ == '__main__':
#    PredictingHastagsForLocations.run()
    PerformanceOfPredictingMethodsByVaryingParameter.run()
    