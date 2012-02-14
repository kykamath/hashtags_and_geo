'''
Created on Feb 14, 2012

@author: kykamath
'''
from analysis import iterateJsonFromFile
from settings import timeUnitWithOccurrencesFile
from datetime import datetime

data = list(iterateJsonFromFile(timeUnitWithOccurrencesFile%'testing'))
for i, data in enumerate(sorted(data, key=lambda d: d['tu'])):
    print i, data['tu'], datetime.fromtimestamp(data['tu'])