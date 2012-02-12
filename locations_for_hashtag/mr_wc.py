'''
Created on Nov 20, 2011

@author: kykamath
'''
from library.mrjobwrapper import ModifiedMRJob
import cjson
from library.geo import getLattice, getLatticeLid


LATTICE_ACCURACY = 0.145

def iterateHashtagObjectInstances(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    point = getLatticeLid(l, LATTICE_ACCURACY)
    if point=='0.0000_0.0000': yield point, l

class MRWC(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
#    def configure_options(self):
#        super(MRWC, self).configure_options()
#        self.add_file_option( '--val', dest='val', default=' sdf ', help='provide initial clusters file')
        
#    def load_options(self, args):
#        """Parse stop_words option."""
#        super(MRWC, self).load_options(args)
#        self.val = self.options.val
#    def mapper(self, key, line):
#        for w in line.split(): yield w, 1
#    def reducer(self, key, values): yield key, [key, sum(list(values))]

    def mapper(self, key, line):
        for k,v in iterateHashtagObjectInstances(line): yield k,v
    def reducer(self, key, values): yield key, [key, list(values)]
    
if __name__ == '__main__':
    MRWC.run()