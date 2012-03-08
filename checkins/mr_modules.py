'''
Created on Feb 1, 2012

@author: kykamath

Method format:
mapper_<input_line>_to_tuple_of_<key>_and_<value>
mapper_final_<input_line>_to_tuple_of_<key>_and_<value>
reducer_tuple_of_<key>_and_iterator_of_<value>_to_tuple_of_<key>_and_<value>
'''
from library.mrjobwrapper import ModifiedMRJob
import cjson
from library.geo import isWithinBoundingBox, getLatticeLid
from collections import defaultdict
from itertools import groupby
from operator import itemgetter

FOURSQUARE_ID = '4sq'

LATTICE_ACCURACY = 0.0001
TIME_UNIT_IN_SECONDS = 6*60*60

#BOUNDARY_ID, BOUNDARY =  'world', [[-90,-180], [90, 180]]
#MINIMUM_NUMBER_OF_CHECKINS_PER_USER = 100
#MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION = 25
#MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION_PER_USER = 3

#BOUNDARY_ID, BOUNDARY =  'usa', [[24.527135,-127.792969], [49.61071,-59.765625]]
#MINIMUM_NUMBER_OF_CHECKINS_PER_USER = 100
#MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION = 25
#MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION_PER_USER = 3

BOUNDARY_ID, BOUNDARY = 'ny', [[40.491, -74.356], [41.181, -72.612]]
MINIMUM_NUMBER_OF_CHECKINS_PER_USER = 100
MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION = 25
MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION_PER_USER = 3

PARAMS_DICT = dict(
                   PARAMS_DICT = True,
                   LATTICE_ACCURACY = LATTICE_ACCURACY,
                   TIME_UNIT_IN_SECONDS = TIME_UNIT_IN_SECONDS,
                   BOUNDARY_ID = BOUNDARY_ID,
                   BOUNDARY = BOUNDARY,
                   MINIMUM_NUMBER_OF_CHECKINS_PER_USER = MINIMUM_NUMBER_OF_CHECKINS_PER_USER,
                   MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION = MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION,
                   MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION_PER_USER = MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION_PER_USER
               )

def getCheckinsObject(line):
    data = cjson.decode(line)
    if data and isWithinBoundingBox(data['l'], BOUNDARY): return data
    
def get_socail_network(user_id):
    if type(user_id)==type(1): return FOURSQUARE_ID
    else: return user_id.split('_')[0]

class CheckinsGraphEdgeWeightMethod:
    @staticmethod
    def jaccard_index(location_1, tuples_of_user_checkin_times_1, location_2, tuples_of_user_checkin_times_2 ):
        set_of_user_1 = set([user for user,_ in tuples_of_user_checkin_times_1])
        set_of_user_2 = set([user for user,_ in tuples_of_user_checkin_times_2])
        return float(len(set_of_user_1.intersection(set_of_user_2)))/float(len(set_of_user_1.union(set_of_user_2)))

class MRCheckins(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(MRCheckins, self).__init__(*args, **kwargs)
        self.userToCheckinsMap = defaultdict(list)
        self.map_from_lid_to_map_from_social_network_to_lid_occurences_count = defaultdict(dict)
        self.map_from_lid_to_tuples_of_user_and_checkin_time = defaultdict(list)
        self.map_from_user_to_tuples_of_lid_and_checkin_time = defaultdict(list)
        self.map_from_lid_to_map_from_neighboring_lid_to_tuples_of_user_and_checkin_time = defaultdict(dict)
    ''' Start: Methods to determine checkin distribution across users
    '''
    def mapCheckinsPerUser(self, key, line):
        if False: yield # I'm a generator!
        checkinObject = getCheckinsObject(line)
        if checkinObject: self.userToCheckinsMap[checkinObject['u']].append([checkinObject['l'], checkinObject['t']])
    def mapCheckinsPerUserFinal(self):
        for u, checkins in self.userToCheckinsMap.iteritems(): yield u, checkins 
    def reducerCheckinsPerUser(self, key, values): 
        checkins = reduce(list.__add__, values, [])
        if len(checkins)>=MINIMUM_NUMBER_OF_CHECKINS_PER_USER: 
            checkins = sorted(checkins, key=lambda t: t[1])
            yield key, {'u': key, 'c': checkins}
    ''' End: Methods to determine checkin distribution across users
    '''
    ''' Start: Methods to determine geo distribution of points across different social networks.
    '''
    def mapper_checkins_json_to_tuple_of_lid_and_map_from_social_network_to_lid_occurences_count(self, key, checkins_json):
        if False: yield # I'm a generator!
        checkin_object = getCheckinsObject(checkins_json)
        if checkin_object:
            social_network = get_socail_network(checkin_object['u'])
            lid = getLatticeLid(checkin_object['l'], accuracy = LATTICE_ACCURACY)
            if social_network not in self.map_from_lid_to_map_from_social_network_to_lid_occurences_count[lid]: 
                self.map_from_lid_to_map_from_social_network_to_lid_occurences_count[lid][social_network]=0.0
            self.map_from_lid_to_map_from_social_network_to_lid_occurences_count[lid][social_network]+=1
    def mapper_final_checkins_json_to_tuple_of_lid_and_map_from_social_network_to_lid_occurences_count(self):
        for lid, map_from_social_network_to_lid_occurences_count in self.map_from_lid_to_map_from_social_network_to_lid_occurences_count.iteritems():
            yield lid, map_from_social_network_to_lid_occurences_count
    def reducer_tuple_of_lid_and_iterator_of_map_from_social_network_to_lid_occurences_count_to_tuple_of_lid_and_map_from_social_network_to_lid_occurences_count(self, lid, iterator_of_map_from_social_network_to_lid_occurences_count):
        aggregated_map_from_social_network_to_lid_occurences_count = defaultdict(float)
        for map_from_social_network_to_lid_occurences_count in iterator_of_map_from_social_network_to_lid_occurences_count:
            for social_network, lid_occurences_count in map_from_social_network_to_lid_occurences_count.iteritems(): 
                aggregated_map_from_social_network_to_lid_occurences_count[social_network]+=lid_occurences_count
        if sum(map_from_social_network_to_lid_occurences_count.values())>=MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION:
            yield lid, {'key': lid, 'distribution': aggregated_map_from_social_network_to_lid_occurences_count}
    ''' End: Methods to determine geo distribution of points across different social networks.
    '''
    ''' Start: Methods to get locations with minimum no. of checkins.
    '''
    def mapper_user_object_to_tuple_of_lid_and_tuple_of_user_and_checkin_time(self, key, user_object):
        if False: yield # I'm a generator!
        for location, checkin_time in user_object['c']: self.map_from_lid_to_tuples_of_user_and_checkin_time[getLatticeLid(location, accuracy = LATTICE_ACCURACY)].append([user_object['u'], checkin_time])
    def mapper_final_user_object_to_tuple_of_lid_and_tuple_of_user_and_checkin_time(self):
        for lid, tuples_of_user_and_checkin_time in self.map_from_lid_to_tuples_of_user_and_checkin_time.iteritems(): 
            yield lid, tuples_of_user_and_checkin_time
    def reducer_tuple_of_lid_and_iterator_of_tuples_of_user_and_checkin_time_to_tuple_of_lid_and_location_object(self, lid, iterator_of_tuples_of_user_and_checkin_time):
        tuples_of_user_and_checkin_time = reduce(list.__add__,  iterator_of_tuples_of_user_and_checkin_time, [])
        tuples_of_user_and_number_of_checkins_at_lid =[(user, len(list(iterator_of_tuples_of_user_and_checkin_time))) for user, iterator_of_tuples_of_user_and_checkin_time in groupby(sorted(tuples_of_user_and_checkin_time, key=itemgetter(0)), key=itemgetter(0))]
        tuples_of_user_and_number_of_checkins_at_lid = filter(lambda (user, number_of_checkins_at_lid): number_of_checkins_at_lid>=MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION_PER_USER, tuples_of_user_and_number_of_checkins_at_lid)
        if tuples_of_user_and_number_of_checkins_at_lid:
            valid_users = zip(*tuples_of_user_and_number_of_checkins_at_lid)[0]
            tuples_of_user_and_checkin_time = filter(lambda (user, _): user in valid_users, tuples_of_user_and_checkin_time)
            if len(tuples_of_user_and_checkin_time)>=MINIMUM_NUMBER_OF_CHECKINS_PER_LOCATION: yield lid, {'l': lid, 'c': tuples_of_user_and_checkin_time}
    ''' End: Methods to get locations with minimum no. of checkins.
    '''
    ''' Start: Methods to get checkins graph.
    '''
    def mapper_location_object_to_tuple_of_lid_and_tuple_of_user_and_checkin_time(self, lid, location_object):
        if False: yield # I'm a generator!
        for user, checkin_time in location_object['c']: self.map_from_user_to_tuples_of_lid_and_checkin_time[user].append([lid, checkin_time])
    def mapper_final_location_object_to_tuple_of_lid_and_tuple_of_user_and_checkin_time(self):
        for user, tuples_of_lid_and_checkin_time in self.map_from_user_to_tuples_of_lid_and_checkin_time.iteritems(): yield user, tuples_of_lid_and_checkin_time
    def reducer_tuple_of_user_and_iterator_of_tuples_of_lid_and_checkin_time_to_tuple_of_user_and_user_object(self, user, iterator_of_tuples_of_lid_and_checkin_time):
        checkins = reduce(list.__add__,  iterator_of_tuples_of_lid_and_checkin_time, [])
        yield user, {'u': user, 'c': checkins}
    def mapper_user_object_to_tuple_of_lid_and_tuple_of_neighborind_lid_and_tuples_of_user_checkins(self, user, user_object):
        if False: yield # I'm a generator!
        lids = list(set([lid for lid, _ in user_object['c']]))
        for lid in lids:
            for neighboring_lid, checkin_time in user_object['c']: 
                if neighboring_lid not in self.map_from_lid_to_map_from_neighboring_lid_to_tuples_of_user_and_checkin_time[lid]:
                    self.map_from_lid_to_map_from_neighboring_lid_to_tuples_of_user_and_checkin_time[lid][neighboring_lid] = []
                self.map_from_lid_to_map_from_neighboring_lid_to_tuples_of_user_and_checkin_time[lid][neighboring_lid].append([user, checkin_time])
    def mapper_final_user_object_to_tuple_of_lid_and_tuple_of_neighborind_lid_and_tuples_of_user_checkins(self):
        for lid, map_from_neighboring_lid_to_tuples_of_user_and_checkin_time in self.map_from_lid_to_map_from_neighboring_lid_to_tuples_of_user_and_checkin_time.iteritems():
            for neighboring_lid, tuples_of_user_and_checkin_time in map_from_neighboring_lid_to_tuples_of_user_and_checkin_time.iteritems():
                yield lid, [neighboring_lid, tuples_of_user_and_checkin_time]
    def reducer_tuple_of_lid_and_iterator_of_tuple_of_neighboring_lid_and_tuples_of_user_and_checkin_time_to_edge_object(self, lid, iterator_of_tuple_of_neighboring_lid_and_tuples_of_user_and_checkin_time):
        map_from_neighboring_lid_to_tuples_of_user_and_checkin_time = defaultdict(list)
        for neighboring_lid, tuples_of_user_and_checkin_time in iterator_of_tuple_of_neighboring_lid_and_tuples_of_user_and_checkin_time:
            map_from_neighboring_lid_to_tuples_of_user_and_checkin_time[neighboring_lid]+=tuples_of_user_and_checkin_time
        for neighboring_lid, tuples_of_user_and_checkin_time in map_from_neighboring_lid_to_tuples_of_user_and_checkin_time.iteritems():
            if lid!=neighboring_lid:
                edge = '__'.join(sorted([lid, neighboring_lid]))
                yield edge, {
                             'e':edge, 
                             'w': CheckinsGraphEdgeWeightMethod.jaccard_index(lid, map_from_neighboring_lid_to_tuples_of_user_and_checkin_time[lid], neighboring_lid, map_from_neighboring_lid_to_tuples_of_user_and_checkin_time[neighboring_lid])
                             }
    def mapper_edge_object_to_tuple_of_edge_and_edge_object(self, edge, edge_object): yield edge, edge_object
    def reducer_tuple_of_edge_and_iterator_of_edge_object_to_tuple_of_edge_and_edge_object(self, edge, iterator_of_edge_object):
        edge_weight = ()
        for edge_object in iterator_of_edge_object: edge_weight = min([edge_weight, edge_object['w']])
        yield edge, { 'e': edge , 'w': edge_weight}
                
    ''' End: Methods to get checkins graph.
    '''
    
    def jobsToGetCheckinsInABoundaryPerUser(self): return [self.mr(mapper=self.mapCheckinsPerUser, mapper_final=self.mapCheckinsPerUserFinal, reducer=self.reducerCheckinsPerUser)]
    def jobs_to_get_geo_distribution_of_points_across_social_networks(self): return [self.mr(
                                                                                             mapper=self.mapper_checkins_json_to_tuple_of_lid_and_map_from_social_network_to_lid_occurences_count, 
                                                                                             mapper_final=self.mapper_final_checkins_json_to_tuple_of_lid_and_map_from_social_network_to_lid_occurences_count, 
                                                                                             reducer=self.reducer_tuple_of_lid_and_iterator_of_map_from_social_network_to_lid_occurences_count_to_tuple_of_lid_and_map_from_social_network_to_lid_occurences_count
                                                                                             )]
    def jobs_to_get_location_objects_with_minumum_checkins_at_both_location_and_users(self):
        return self.jobsToGetCheckinsInABoundaryPerUser() + \
                [self.mr(
                     mapper=self.mapper_user_object_to_tuple_of_lid_and_tuple_of_user_and_checkin_time, 
                     mapper_final=self.mapper_final_user_object_to_tuple_of_lid_and_tuple_of_user_and_checkin_time, 
                     reducer=self.reducer_tuple_of_lid_and_iterator_of_tuples_of_user_and_checkin_time_to_tuple_of_lid_and_location_object
                     )]
    def jobs_to_get_checkins_graph(self):
        return self.jobs_to_get_location_objects_with_minumum_checkins_at_both_location_and_users() + \
            [self.mr(
                     mapper=self.mapper_location_object_to_tuple_of_lid_and_tuple_of_user_and_checkin_time, 
                     mapper_final=self.mapper_final_location_object_to_tuple_of_lid_and_tuple_of_user_and_checkin_time, 
                     reducer=self.reducer_tuple_of_user_and_iterator_of_tuples_of_lid_and_checkin_time_to_tuple_of_user_and_user_object
                     )] + \
            [self.mr(
                     mapper=self.mapper_user_object_to_tuple_of_lid_and_tuple_of_neighborind_lid_and_tuples_of_user_checkins, 
                     mapper_final=self.mapper_final_user_object_to_tuple_of_lid_and_tuple_of_neighborind_lid_and_tuples_of_user_checkins, 
                     reducer=self.reducer_tuple_of_lid_and_iterator_of_tuple_of_neighboring_lid_and_tuples_of_user_and_checkin_time_to_edge_object
                     )] + \
            [self.mr(
                     mapper=self.mapper_edge_object_to_tuple_of_edge_and_edge_object, 
                     reducer=self.reducer_tuple_of_edge_and_iterator_of_edge_object_to_tuple_of_edge_and_edge_object
                     )]
    def steps(self):
        pass
#        return self.jobsToGetCheckinsInABoundaryPerUser()
#        return self.jobs_to_get_geo_distribution_of_points_across_social_networks()
        return self.jobs_to_get_location_objects_with_minumum_checkins_at_both_location_and_users()
#        return self.jobs_to_get_checkins_graph()
    
if __name__ == '__main__':
    MRCheckins.run()