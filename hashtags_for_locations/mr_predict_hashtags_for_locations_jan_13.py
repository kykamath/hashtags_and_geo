'''
Created on Jan 22, 2013

@author: krishnakamath
'''
from collections import defaultdict
from itertools import chain
from library.mrjobwrapper import ModifiedMRJob
from operator import itemgetter
import numpy as np

class HashtagsByModelsByLocations(ModifiedMRJob):
    def __init__(self, num_of_hashtags=5, *args, **kwargs):
        super(HashtagsByModelsByLocations, self).__init__(*args, **kwargs)
        self.num_of_hashtags = num_of_hashtags
    def get_valid_locations(self, input_data):
        so_valid_locations = set(input_data['mf_location_to_ideal_hashtags_rank'])
        for model_id, locations in input_data['mf_model_id_to_mf_location_to_hashtags_ranked_by_model'].iteritems():
            so_valid_locations = so_valid_locations.intersection(locations.keys())
        return list(so_valid_locations)
    def get_ideal_hashtags(self, mf_location_to_ideal_hashtags_rank, valid_locations, output_data):
        for location, ideal_hashtags_rank in mf_location_to_ideal_hashtags_rank.iteritems():
            if location in valid_locations:
                ideal_hashtags_rank.sort(key=itemgetter(1), reverse=True)
                if location not in output_data['locations']: output_data['locations'][location] = {}
                output_data['locations'][location]['ideal_ranking'] = \
                                                    ideal_hashtags_rank[:self.num_of_hashtags]
    def get_model_predicted_hashtags(
                                     self,
                                     mf_model_id_to_mf_location_to_hashtags_ranked_by_model,
                                     valid_locations,
                                     output_data
                                     ):
        for model_id, mf_location_to_hashtags_ranked_by_model in \
                mf_model_id_to_mf_location_to_hashtags_ranked_by_model.iteritems():
            for location, hashtags_ranked_by_model in mf_location_to_hashtags_ranked_by_model.iteritems():
                if location in valid_locations:
                    hashtags_ranked_by_model.sort(key=itemgetter(1), reverse=True)
                    hashtags = zip(*hashtags_ranked_by_model[:self.num_of_hashtags])[0]
                    if location not in output_data['locations']: output_data['locations'][location] = {}
                    output_data['locations'][location][model_id] = list(hashtags)
    def filter_locations(self, output_data):
        for location, mf_model_id_to_hashtags in output_data['locations'].items()[:]:
            if len(mf_model_id_to_hashtags['ideal_ranking']) < self.num_of_hashtags:
                del output_data['locations'][location]
    def mapper(self, key, input_data):
        output_data = {'locations': {}, 'tu': input_data['tu']}
        valid_locations = self.get_valid_locations(input_data)
        self.get_ideal_hashtags(input_data['mf_location_to_ideal_hashtags_rank'], valid_locations, output_data)
        self.get_model_predicted_hashtags(
                                          input_data['mf_model_id_to_mf_location_to_hashtags_ranked_by_model'],
                                          valid_locations,
                                          output_data
                                        )
        self.filter_locations(output_data)
        if output_data['locations']: yield input_data['tu'], output_data
        
class EvaluationMetric(object):
    ID_ACCURACY = 'accuracy'
    ID_IMPACT = 'impact'
    @staticmethod
    def accuracy(best_hashtags, predicted_hashtags, num_of_hashtags):
        return len(set(best_hashtags).intersection(set(predicted_hashtags))) / float(len(best_hashtags))
    @staticmethod
    def impact(best_hashtags, predicted_hashtags, hashtags_dist):
        total_perct_for_best_hashtags = sum([hashtags_dist.get(h, 0.0) for h in best_hashtags])
        total_perct_for_predicted_hashtags = sum([hashtags_dist.get(h, 0.0) for h in predicted_hashtags])
        return total_perct_for_predicted_hashtags / float(total_perct_for_best_hashtags)

class ModelPerformance(ModifiedMRJob):
    NUM_OF_HASHTAGS = 10
    def __init__(self, *args, **kwargs):
        super(ModelPerformance, self).__init__(*args, **kwargs)
        self.hashtags_by_models_by_locations = HashtagsByModelsByLocations(num_of_hashtags=1)
        self.mf_model_id_to_mf_metric_to_values = {}
    def get_ideal_hashtags(self, mf_location_to_ideal_hashtags_rank, output_data):
        for location, ideal_hashtags_rank in mf_location_to_ideal_hashtags_rank.iteritems():
            ideal_hashtags_rank.sort(key=itemgetter(1), reverse=True)
            if location not in output_data['locations']: output_data['locations'][location] = {}
            output_data['locations'][location]['ideal_ranking'] = ideal_hashtags_rank[:ModelPerformance.NUM_OF_HASHTAGS]
    def get_model_predicted_hashtags(
                                     self,
                                     mf_model_id_to_mf_location_to_hashtags_ranked_by_model,
                                     output_data
                                     ):
        for model_id, mf_location_to_hashtags_ranked_by_model in \
                mf_model_id_to_mf_location_to_hashtags_ranked_by_model.iteritems():
            for location, hashtags_ranked_by_model in mf_location_to_hashtags_ranked_by_model.iteritems():
                hashtags_ranked_by_model.sort(key=itemgetter(1), reverse=True)
                hashtags = zip(*hashtags_ranked_by_model[:ModelPerformance.NUM_OF_HASHTAGS])[0]
                if location not in output_data['locations']: output_data['locations'][location] = {}
                output_data['locations'][location][model_id] = list(hashtags)
    def process_initial_data(self, input_data):
        output_data = {'locations': {}, 'tu': input_data['tu']}
        self.get_ideal_hashtags(input_data['mf_location_to_ideal_hashtags_rank'], output_data)
        self.get_model_predicted_hashtags(
                                          input_data['mf_model_id_to_mf_location_to_hashtags_ranked_by_model'],
                                          output_data
                                        )
        return output_data
    def mapper(self, key, input_data):
        if False: yield # I'm a generator!
        data = self.process_initial_data(input_data)
        ltuo_location_and_mf_model_id_to_hashtags = data['locations'].items()
        ltuo_location_and_mf_model_id_to_hashtags = filter(
                                                           lambda (_,d): 'ideal_ranking' in d, 
                                                           ltuo_location_and_mf_model_id_to_hashtags
                                                           )
        for location, mf_model_id_to_hashtags in ltuo_location_and_mf_model_id_to_hashtags:
            best_hashtags = set(zip(*mf_model_id_to_hashtags['ideal_ranking'])[0])
            mf_hashtag_to_score = dict(mf_model_id_to_hashtags['ideal_ranking'])
            for model_id, hashtags in mf_model_id_to_hashtags.iteritems():
                if model_id not in 'ideal_ranking':
                    if model_id not in self.mf_model_id_to_mf_metric_to_values:
                        self.mf_model_id_to_mf_metric_to_values[model_id] = defaultdict(list)
                    self.mf_model_id_to_mf_metric_to_values[model_id][EvaluationMetric.ID_ACCURACY].append(
                                                                EvaluationMetric.accuracy(best_hashtags, hashtags, None)
                                                            )
                    self.mf_model_id_to_mf_metric_to_values[model_id][EvaluationMetric.ID_IMPACT].append(
                                                EvaluationMetric.impact(best_hashtags, hashtags, mf_hashtag_to_score)
                    )
    def mapper_final(self):
        for model_id, mf_metric_to_values in self.mf_model_id_to_mf_metric_to_values.iteritems():
            yield model_id, mf_metric_to_values
    def reducer(self, model_id, it_mf_metric_to_values):
        mf_metric_to_values = list(it_mf_metric_to_values)
        accuracy = np.mean(list(chain(*map(itemgetter(EvaluationMetric.ID_ACCURACY), mf_metric_to_values))))
        impact = np.mean(list(chain(*map(itemgetter(EvaluationMetric.ID_IMPACT), mf_metric_to_values))))
        yield model_id, {
                             'model_id': model_id,
                             EvaluationMetric.ID_ACCURACY: accuracy,
                             EvaluationMetric.ID_IMPACT:impact
                        }

if __name__ == '__main__':
#    HashtagsByModelsByLocations.run()
    ModelPerformance.run()
