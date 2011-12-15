'''
Created on Dec 15, 2011

@author: kykamath
'''
import sys
sys.path.append('../')
from library.mrjobwrapper import runMRJob
from library.file_io import FileIO
from graph_analysis.mr_modules import MRGraph
from graph_analysis.settings import hdfsInputFolder, epochGraphsFile

def temp_analysis():
    for d in FileIO.iterateJsonFromFile(epochGraphsFile%('%s_%s'%timeRange)+'_bak'):
        print dict(d).keys()

def getInputFiles(months, folderType='/'): return [hdfsInputFolder+folderType+'/'+str(m) for m in months] 
def mr_task(timeRange, dataType, mrOutputFolder=None):
#    runMRJob(MRGraph, hashtagsFile%('%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), dataType), jobconf={'mapred.reduce.tasks':160})
    runMRJob(MRGraph, epochGraphsFile%('%s_%s'%timeRange), getInputFiles(range(timeRange[0], timeRange[1]+1), dataType), jobconf={'mapred.reduce.tasks':160})   

if __name__ == '__main__':
    timeRange = (2,3)
    dataType = 'world'
    mr_task(timeRange, dataType)
#    temp_analysis()