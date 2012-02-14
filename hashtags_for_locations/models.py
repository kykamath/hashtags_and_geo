'''
Created on Feb 14, 2012

@author: kykamath
'''
from analysis import iterateJsonFromFile
from settings import timeUnitWithOccurrencesFile
from datetime import datetime, timedelta
from mr_analysis import TIME_UNIT_IN_SECONDS
import time
from collections import defaultdict

class PropagationsInInterval:
    def __init__(self, startTime, interval):
        self.startTime, self.interval = startTime, interval
        self.occurrances = defaultdict(list)
    def updateOccurrences(self, occurrances):
        for h, loc, t in occurrances: self.occurrances[h].append([loc, t])

historyTimeInterval = timedelta(seconds=30*60)
predictionTimeInterval = timedelta(seconds=120*60)

startTime, endTime, outputFolder = datetime(2011, 11, 1), datetime(2011, 12, 31), 'testing'
currentTime = startTime
timeUnitDelta = timedelta(seconds=TIME_UNIT_IN_SECONDS)
timeUnitsToDataMap = dict([(d['tu'], d) for d in iterateJsonFromFile(timeUnitWithOccurrencesFile%outputFolder)])
while currentTime<endTime:
    currentTimeEpoch = time.mktime(currentTime.timetuple())
    timeUnitObject = timeUnitsToDataMap.get(currentTimeEpoch, None)
    if timeUnitObject: print currentTime, len(timeUnitObject['oc'])
    else: print currentTime, 0
    currentTime+=timeUnitDelta
    
#timeUnitsToDataMap = [(d['tu'], d) for d in iterateJsonFromFile(timeUnitWithOccurrencesFile%outputFolder)]
#for i, data in enumerate(sorted(data, key=lambda d: d['tu'])):
#    print i, data['tu'], datetime.fromtimestamp(data['tu'])