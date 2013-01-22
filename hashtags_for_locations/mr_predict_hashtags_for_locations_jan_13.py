'''
Created on Jan 22, 2013

@author: krishnakamath
'''
from library.mrjobwrapper import ModifiedMRJob
from operator import itemgetter

class HashtagsByModelsByLocations(ModifiedMRJob):
    NUM_OF_HASHTAGS = 4
    def get_valid_locations(self, input_data):
        so_valid_locations = set(input_data['mf_location_to_ideal_hashtags_rank'])
        for model_id, locations in input_data['mf_model_id_to_mf_location_to_hashtags_ranked_by_model'].iteritems():
            so_valid_locations = so_valid_locations.intersection(locations.keys())
        return list(so_valid_locations)
    def get_ideal_hashtags(self, mf_location_to_ideal_hashtags_rank, valid_locations, output_data):
        for location, ideal_hashtags_rank in mf_location_to_ideal_hashtags_rank.iteritems():
            if location in valid_locations:
#            if len(ideal_hashtags_rank) >= HashtagsByModelsByLocations.NUM_OF_HASHTAGS:
                ideal_hashtags_rank.sort(key=itemgetter(1), reverse=True)
                if location not in output_data['locations']: output_data['locations'][location] = {}
                output_data['locations'][location]['ideal_ranking'] = \
                                                    ideal_hashtags_rank[:HashtagsByModelsByLocations.NUM_OF_HASHTAGS]
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
#                if len(hashtags_ranked_by_model) >= HashtagsByModelsByLocations.NUM_OF_HASHTAGS:
                    hashtags_ranked_by_model.sort(key=itemgetter(1), reverse=True)
                    hashtags = zip(*hashtags_ranked_by_model[:HashtagsByModelsByLocations.NUM_OF_HASHTAGS])[0]
                    if location not in output_data['locations']: output_data['locations'][location] = {}
                    output_data['locations'][location][model_id] = list(hashtags)
    def filter_locations(self, output_data):
        for location, mf_model_id_to_hashtags in output_data['locations'].items()[:]:
            if len(mf_model_id_to_hashtags['ideal_ranking']) < HashtagsByModelsByLocations.NUM_OF_HASHTAGS:
                del output_data['locations'][location]
#            else: 
#                for model_id, hashtags in mf_model_id_to_hashtags.iteritems():
#                    print model_id, len(hashtags),
#                print 
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
        if output_data['locations']: yield key, output_data

if __name__ == '__main__':
    HashtagsByModelsByLocations.run()
