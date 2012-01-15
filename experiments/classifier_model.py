'''
Created on Jan 14, 2012

@author: kykamath
'''
from settings import targetSelectionRegressionClassifiersFolder, hashtagsFile
from library.file_io import FileIO
from experiments.models import Hashtag
from experiments.mr_area_analysis import HashtagsClassifier
from itertools import groupby
from operator import itemgetter
from collections import defaultdict

class TargetSelectionRegressionClassifier:
    def __init__(self, decisionTimeUnit):
        self.decisionTimeUnit = decisionTimeUnit
#        self.predictingLattice = predictingLattice
        self.modelFile = targetSelectionRegressionClassifiersFolder%(self.decisionTimeUnit, self.predictingLattice)+'model.pkl'
def build():
    documents = []
    for h in FileIO.iterateJsonFromFile(hashtagsFile%('training_world','%s_%s'%(2,11))): 
        hashtag, document = Hashtag(h), []
        if hashtag.isValidObject():
            for timeUnit, occs in enumerate(hashtag.getOccrancesEveryTimeWindowIterator(HashtagsClassifier.CLASSIFIER_TIME_UNIT_IN_SECONDS)):
                occs = sorted(occs, key=itemgetter(0))
                document.append([timeUnit, [(k, len(list(i))) for k, i in groupby(occs, key=itemgetter(0))]])
            if document:
                print document
                print zip(*document)[0]
#            documents.append(document)
#        print documents
#        exit()
if __name__ == '__main__':
    build()