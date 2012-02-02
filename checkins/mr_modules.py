'''
Created on Feb 1, 2012

@author: kykamath
'''
from library.mrjobwrapper import ModifiedMRJob
import cjson
from library.geo import isWithinBoundingBox
from collections import defaultdict

LATTICE_ACCURACY = 0.145

BOUNDARY = [[-90,-180], [90, 180]] # World
#BOUNDARY = [[24.527135,-127.792969], [49.61071,-59.765625]] #US
#BOUNDARY = [[40.491, -74.356], [41.181, -72.612]] #NY

MINIMUM_NUMBER_OF_CHECKINS_PER_USER = 50

def getCheckinsObject(line):
    data = cjson.decode(line)
    if data and isWithinBoundingBox(data['l'], BOUNDARY): return data

class MRCheckins(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(MRCheckins, self).__init__(*args, **kwargs)
        self.userToCheckinsMap = defaultdict(list)
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
            yield key, {'u': key, 'c': len(checkins)}
    
    def jobsToGetCheckinsInABoundaryPerUser(self): return [self.mr(mapper=self.mapCheckinsPerUser, mapper_final=self.mapCheckinsPerUserFinal, reducer=self.reducerCheckinsPerUser)]
    def steps(self):
        pass
        return self.jobsToGetCheckinsInABoundaryPerUser()
    
if __name__ == '__main__':
    MRCheckins.run()