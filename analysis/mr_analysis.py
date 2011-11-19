'''
Created on Nov 19, 2011

@author: kykamath
'''
from library.twitter import getDateTimeObjectFromTweetTimestamp
from library.mrjobwrapper import ModifiedMRJob
import cjson, time, datetime
from collections import defaultdict

MIN_HASHTAG_OCCURENCES = 25
HASHTAG_STARTING_WINDOW = datetime.datetime(2011, 2, 1)
HASHTAG_ENDING_WINDOW = datetime.datetime(2011, 11, 30)

def iterateHashtagObjectInstances(line):
    data = cjson.decode(line)
    l = data['geo']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    for h in data['h']: yield h, [l, t]

class MRAnalysis(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(MRAnalysis, self).__init__(*args, **kwargs)
        self.hashtags = defaultdict(list)
        
    ''' Start: Methods to get hashtag objects
    '''
    def parse_hashtag_objects(self, key, line):
        if False: yield # I'm a generator!
        for h, d in iterateHashtagObjectInstances(line): self.hashtags[h].append(d)
    def parse_hashtag_objects_final(self):
        for h, instances in self.hashtags.iteritems(): # e = earliest, l = latest
            yield h, {'oc': instances, 'e': min(instances, key=lambda t: t[1]), 'l': max(instances, key=lambda t: t[1])}
    def combine_hashtag_instances(self, key, values):
        occurences = []
        e, l = None, None
        for instances in values:
            if e==None or e<instances['e'][1]: e = instances['e']
            if l==None or l>instances['l'][1]: l = instances['l']
            occurences+=instances['oc']
        numberOfInstances=len(occurences)
        if numberOfInstances>=MIN_HASHTAG_OCCURENCES and \
            datetime.datetime.fromtimestamp(e[1])>=HASHTAG_STARTING_WINDOW and \
            datetime.datetime.fromtimestamp(l[1])<=HASHTAG_ENDING_WINDOW:
                yield key, {'h': key, 't': numberOfInstances, 'e':e, 'l':l, 'oc': occurences}
    ''' End: Methods to get hashtag objects
    '''
    
    def jobsToGetHastagObjects(self): return [self.mr(mapper=self.parse_hashtag_objects, mapper_final=self.parse_hashtag_objects_final, reducer=self.combine_hashtag_instances)]
    
    def steps(self):
        return self.jobsToGetHastagObjects() + self.jobsToCountNumberOfKeys()

if __name__ == '__main__':
    MRAnalysis.run()