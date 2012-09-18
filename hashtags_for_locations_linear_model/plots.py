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
from settings import f_significant_nei_utm_ids
from settings import fld_google_drive_data_analysis
import matplotlib

class DataAnalysisPlots(object):
    @staticmethod
    def utm_ids_on_map():
        ''' Plots utm ids on world map. The color indicates the
        log(total_hashtag_count)
        '''
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.png'
        ltuo_point_and_total_hashtag_count = []
        for utm_object in FileIO.iterateJsonFromFile(f_hashtags_by_utm_id, remove_params_dict=True):
            point = UTMConverter.getLatLongUTMIdInLatLongForm(utm_object['utm_id'])
            total_hashtag_count = log(utm_object['total_hashtag_count'])
            ltuo_point_and_total_hashtag_count.append((point, total_hashtag_count))
        points, total_hashtag_counts = zip(*sorted(ltuo_point_and_total_hashtag_count, key=itemgetter(1)))
        plotPointsOnWorldMap(points,
                             blueMarble=False,
                             bkcolor='#CFCFCF',
                             c=total_hashtag_counts,
                             cmap=matplotlib.cm.cool,
                             lw = 0,
                             alpha=1.)
        
        savefig(output_file)
    @staticmethod
    def significant_nei_utm_ids():
        output_folder = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'/%s.png'
        for i, data in enumerate(FileIO.iterateJsonFromFile(f_significant_nei_utm_ids, remove_params_dict=True)):
            utm_lat_long = UTMConverter.getLatLongUTMIdInLatLongForm(data['utm_id'])
            nei_utm_lat_longs = map(
                              lambda nei_utm_id: UTMConverter.getLatLongUTMIdInLatLongForm(nei_utm_id),
                              data['nei_utm_ids']
                            )
            if nei_utm_lat_longs:
                output_file = output_folder%('%s_%s'%(utm_lat_long))
                plotPointsOnWorldMap(nei_utm_lat_longs,
                                     blueMarble=False,
                                     bkcolor='#CFCFCF',
                                     lw = 0,
                                     color = '#EA00FF',
                                     alpha=1.)
                _, m = plotPointsOnWorldMap([utm_lat_long],
                                     blueMarble=False,
                                     bkcolor='#CFCFCF',
                                     lw = 0,
                                     color = '#2BFF00',
                                     s = 40,
                                     returnBaseMapObject=True,
                                     alpha=1.)
                for nei_utm_lat_long in nei_utm_lat_longs:
                    m.drawgreatcircle(utm_lat_long[1],
                                      utm_lat_long[0],
                                      nei_utm_lat_long[1],
                                      nei_utm_lat_long[0],
                                      color='#FFA600',
                                      lw=1.5,
                                      alpha=1.0)
                print 'Saving %s'%(i+1)
                savefig(output_file)
#                exit()
    @staticmethod
    def run():
#        DataAnalysisPlots.utm_ids_on_map()
        DataAnalysisPlots.significant_nei_utm_ids()
        
if __name__ == '__main__':
    DataAnalysisPlots.run()