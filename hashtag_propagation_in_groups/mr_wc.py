'''
Created on Oct 12, 2012

@author: krishnakamath
'''
from library.mrjobwrapper import ModifiedMRJob

class MRWordCounter(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def mapper(self, key, line):
        for word in line.split(): yield word, 1

    def reducer(self, word, occurrences): yield word, sum(occurrences)

if __name__ == '__main__':
    MRWordCounter.run()