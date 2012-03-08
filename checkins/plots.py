'''
Created on Mar 7, 2012

@author: kykamath
'''
from checkins.settings import lidsToDistributionInSocialNetworksMapFile
from checkins.mr_modules import BOUNDARY_ID
from checkins.analysis import iterateJsonFromFile
class DataAnalysis:
    @staticmethod
    def plot_geo_distribution_in_social_networks():
        total_checkins = 0.0
        for i, data in enumerate(iterateJsonFromFile(lidsToDistributionInSocialNetworksMapFile%BOUNDARY_ID)):
#            if len(data['distribution']) > 1:
            print i,total_checkins, sum(data['distribution'].values())
            total_checkins+=sum(data['distribution'].values())
#            if i==100: break;
        print total_checkins
    @staticmethod
    def run():
        DataAnalysis.plot_geo_distribution_in_social_networks()
    
if __name__ == '__main__':
    DataAnalysis.run()