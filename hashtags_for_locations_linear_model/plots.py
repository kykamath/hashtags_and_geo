'''
Created on Sep 11, 2012

@author: krishnakamath
'''
from operator import itemgetter
from library.file_io import FileIO
from library.classes import GeneralMethods
from library.geo import UTMConverter
from library.geo import plotPointsOnWorldMap
from library.plotting import savefig
from math import log
from mr_analysis import ACCURACY
from settings import f_hashtags_by_utm_id
from settings import fld_google_drive_data_analysis
import matplotlib

class DataAnalysis(object):
    @staticmethod
    def utm_ids_on_map():
        ''' Plots utm ids on world map. The color indicates the
        log(total_hashtag_count)
        '''
        output_file = \
            fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.png'
        ltuo_point_and_total_hashtag_count = []
        for utm_object in FileIO.iterateJsonFromFile(f_hashtags_by_utm_id,
                                                     remove_params_dict=True):
            point = UTMConverter.getLatLongFromUTMId(utm_object['utm_id'], 
                                                     ACCURACY)
            total_hashtag_count = log(utm_object['total_hashtag_count'])
            ltuo_point_and_total_hashtag_count.append((point,
                                                       total_hashtag_count))
        points, total_hashtag_counts = \
            zip(*sorted(ltuo_point_and_total_hashtag_count, key=itemgetter(1)))
        plotPointsOnWorldMap(points,
                             blueMarble=False,
                             bkcolor='#CFCFCF',
                             c=total_hashtag_counts,
                             cmap=matplotlib.cm.cool,
                             lw = 0,
                             alpha=1.)
        savefig(output_file)
            
    @staticmethod
    def run():
        DataAnalysis.utm_ids_on_map()
        
if __name__ == '__main__':
    DataAnalysis.run()