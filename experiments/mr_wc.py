'''
Created on Nov 20, 2011

@author: kykamath
'''
from library.mrjobwrapper import ModifiedMRJob

class MRWC(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def configure_options(self):
        super(MRWC, self).configure_options()
        self.add_file_option( '--val', dest='val', default=' sdf ', help='provide initial clusters file')
        
    def load_options(self, args):
        """Parse stop_words option."""
        super(MRWC, self).load_options(args)
        self.val = self.options.val
    def mapper(self, key, line):
        for w in line.split(): yield w, 1
    def reducer(self, key, values): yield key, [key, sum(list(values))]
    
if __name__ == '__main__':
    MRWC.run()