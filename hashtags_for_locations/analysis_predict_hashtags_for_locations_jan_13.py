'''
Created on Jan 22, 2013

@author: krishnakamath
'''
from library.file_io import FileIO

f_name = '/mnt/chevron/kykamath/data_from_dfs/geo/hashtags//2011-09-01_2011-11-01/360_120/100/linear_regression'
for data in FileIO.iterateJsonFromFile(f_name, remove_params_dict=True):
    print data.keys()