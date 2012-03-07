'''
Created on Mar 7, 2012

@author: kykamath
'''
from library.file_io import FileIO
from checkins.settings import lidsToDistributionInSocialNetworksMapFile
class DataAnalysis:
    @staticmethod
    def plot_geo_distribution_in_social_networks():
        for i, data in enumerate(FileIO.iterateJsonFromFile(lidsToDistributionInSocialNetworksMapFile)):
#            if len(data['distribution']) > 1:
            print i, data
        
    @staticmethod
    def run():
        DataAnalysis.plot_geo_distribution_in_social_networks()
    
if __name__ == '__main__':
    DataAnalysis.run()