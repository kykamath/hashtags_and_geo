'''
Created on Sept 9, 2012

@author: kykamath
'''
from dateutil.relativedelta import relativedelta
from library.classes import GeneralMethods
from library.file_io import FileIO
from library.mrjobwrapper import runMRJob
from library.mrjobwrapper import runMRJobAndYieldResult
from library.r_helper import R_Helper
from datetime import datetime
from mr_analysis import HashtagsByUTMId
from mr_analysis import HashtagsDistributionInUTM
from mr_analysis import HashtagsExtractor
from mr_analysis import HastagsWithUTMIdObject
from mr_analysis import PARAMS_DICT
from mr_analysis import TweetStats
from operator import itemgetter
from pprint import pprint
from settings import f_hashtags_by_utm_id
from settings import f_hashtag_dist_by_accuracy
from settings import f_hashtags_extractor
from settings import f_hashtags_with_utm_id_object
from settings import f_tweet_stats
from settings import fld_google_drive_data_analysis
from settings import hdfs_input_folder
import rpy2.rlike.container as rlc
import rpy2.robjects as robjects
import random
import time

def getInputFiles(startTime, endTime, folderType='world'):
    current=startTime
    while current<=endTime:
        input_file = hdfs_input_folder%folderType+'%s_%s'%(current.year,
                                                           current.month)
        print input_file
        yield input_file
        current+=relativedelta(months=1)   

class MRAnalysis(object):
    @staticmethod
    def run_job(mr_class,
                output_file,
                input_files_start_time,
                input_files_end_time):
        PARAMS_DICT['input_files_start_time'] = \
            time.mktime(input_files_start_time.timetuple())
        PARAMS_DICT['input_files_end_time'] = \
            time.mktime(input_files_end_time.timetuple())
        print 'Running map reduce with the following params:', \
                pprint(PARAMS_DICT)
        runMRJob(mr_class,
                 output_file,
                 getInputFiles(input_files_start_time, input_files_end_time),
                 jobconf={'mapred.reduce.tasks':500})
        FileIO.writeToFileAsJson(PARAMS_DICT, output_file)
    @staticmethod
    def tweet_stats(input_files_start_time, input_files_end_time):
        mr_class = TweetStats
        output_file = f_tweet_stats
        runMRJob(mr_class,
                 output_file,
                 getInputFiles(input_files_start_time, input_files_end_time),
                 mrJobClassParams = {'job_id': 'as'},
                 jobconf={'mapred.reduce.tasks':300})
        FileIO.writeToFileAsJson(PARAMS_DICT, output_file)
    @staticmethod
    def hashtags_extractor(input_files_start_time, input_files_end_time):
        mr_class = HashtagsExtractor
        output_file = f_hashtags_extractor
        MRAnalysis.run_job(mr_class,
                           output_file,
                           input_files_start_time,
                           input_files_end_time)
    @staticmethod
    def hashtag_dist_by_accuracy(input_files_start_time, input_files_end_time):
        mr_class = HashtagsDistributionInUTM
        output_file = f_hashtag_dist_by_accuracy
        MRAnalysis.run_job(mr_class,
                           output_file,
                           input_files_start_time,
                           input_files_end_time)
    @staticmethod
    def hashtags_by_utm_id(input_files_start_time, input_files_end_time):
        mr_class = HashtagsByUTMId
        output_file = f_hashtags_by_utm_id
        MRAnalysis.run_job(mr_class,
                           output_file,
                           input_files_start_time,
                           input_files_end_time)
    @staticmethod
    def hashtags_with_utm_id_object(input_files_start_time,
                                    input_files_end_time):
        mr_class = HastagsWithUTMIdObject
        output_file = f_hashtags_with_utm_id_object
        MRAnalysis.run_job(mr_class,
                           output_file,
                           input_files_start_time,
                           input_files_end_time)
    
    @staticmethod
    def run():
#        input_files_start_time, input_files_end_time = \
#                        datetime(2011, 2, 1), datetime(2011, 4, 30)
        input_files_start_time, input_files_end_time = \
                                datetime(2011, 2, 1), datetime(2012, 8, 31)
#        MRAnalysis.tweet_stats(input_files_start_time, input_files_end_time)
#        MRAnalysis.hashtags_extractor(input_files_start_time,
#                                      input_files_end_time)
#        MRAnalysis.hashtag_dist_by_accuracy(input_files_start_time,
#                                            input_files_end_time)
        MRAnalysis.hashtags_by_utm_id(input_files_start_time,
                                      input_files_end_time)
#        MRAnalysis.hashtags_with_utm_id_object(input_files_start_time,
#                                               input_files_end_time)
    
class GeneralAnalysis(object):
    @staticmethod
    def print_dense_utm_ids():
        ''' Prints list of dense utm_ids.
        '''
        print [utm_object['utm_id'] 
               for utm_object in FileIO.iterateJsonFromFile(
                                                    f_hashtags_by_utm_id,
                                                    remove_params_dict=True)]
    @staticmethod
    def utm_object_analysis():
        ltuo_utm_id_and_num_of_neighbors = []
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.df'
        for utm_object in FileIO.iterateJsonFromFile(f_hashtags_by_utm_id, True):
            ltuo_utm_id_and_num_of_neighbors.append([utm_object['utm_id'],
                                                     len(utm_object['mf_nei_utm_id_to_common_h_count'])])
        utm_ids, num_of_neighbors = zip(*ltuo_utm_id_and_num_of_neighbors)
        od = rlc.OrdDict([
                          ('utm_ids', robjects.StrVector(utm_ids)),
                          ('num_of_neighbors', robjects.FloatVector(num_of_neighbors))
                        ])
        df = robjects.DataFrame(od)
        FileIO.createDirectoryForFile(output_file)
        print 'Saving df to: ', output_file
        df.to_csvfile(output_file)
    @staticmethod
    def determine_influential_variables():
        x = robjects.FloatVector([random.random() for i in range(10)])
        y = robjects.FloatVector([random.random() for i in range(10)])
#        x1 =  y + robjects.r.rnorm(10)
        od = rlc.OrdDict([('x', x), ('y', y)])
        df = robjects.DataFrame(od)
        df.rownames = 'a b c d e f g h i j'.split()
#        print x.r_repr()
#        print x.r_repr()
#        print len(df), df[0]
#        print df.r_repr()
        print df.nrow
        print df
        
#        robjects.r.png('file.png')
#        robjects.r.plot(x,y)
#        robjects.r['dev.off']()
#        print df.rx(robjects.IntVector(range(1,100)), 'x')
    @staticmethod
    def blah_analysis():
        def get_utm_vectors():
            so_hashtags = set()
            for utm_object in \
                    FileIO.iterateJsonFromFile(f_hashtags_by_utm_id, True):
                for hashtag, count \
                        in utm_object['mf_hashtag_to_count'].iteritems():
                    if hashtag!='total_num_of_occurrences': so_hashtags.add(hashtag)
            hashtags, ltuo_utm_id_and_vector = sorted(list(so_hashtags)), []
            print len(hashtags)
            for i, utm_object in \
                    enumerate(FileIO.iterateJsonFromFile(f_hashtags_by_utm_id,
                                                         True)):
                print i, utm_object['utm_id']
                utm_id_vector = \
                    map(lambda hashtag: 
                            utm_object['mf_hashtag_to_count'].get(hashtag, 0.0),
                        hashtags
                        )
                ltuo_utm_id_and_vector.append((utm_object['utm_id'], 
                                               robjects.FloatVector(utm_id_vector)))
            od = rlc.OrdDict(sorted(ltuo_utm_id_and_vector, key=itemgetter(0)))
            df_utm_vectors = robjects.DataFrame(od)
            return df_utm_vectors
        df_utm_vectors = get_utm_vectors()
        utm_ids = df_utm_vectors.colnames
        for utm_id in utm_ids:
            prediction_variable = utm_id
            predictor_variables = list(df_utm_vectors.colnames)
            print len(utm_ids), len(predictor_variables)
            predictor_variables.remove(utm_id)
            print len(utm_ids), len(predictor_variables)
            print R_Helper.variable_selection_using_backward_elimination(
                                                       df_utm_vectors,
                                                       prediction_variable,
                                                       predictor_variables,
                                                       debug=True
                                                    )
            exit()
#        print len(utm_id_vector)
#            print utm_object['utm_id'], sum(utm_id_vector), \
#                    utm_object['total_hashtag_count']
    @staticmethod
    def test_r():
        od = rlc.OrdDict([('value', robjects.IntVector((1,2,3))),
                      ('letter', robjects.StrVector(('x', 'y', 'z')))])
        dataf = robjects.DataFrame(od)
        print(dataf.colnames)
        plot = robjects.r.plot
        rnorm = robjects.r.rnorm
#        x = robjects.IntVector(range(10))
#        y = x.ro + rnorm(10)
#        print x.r_repr()
        plot(rnorm(100))
##        print rnorm(100)
        
    @staticmethod
    def run():
#        GeneralAnalysis.print_dense_utm_ids()
#        GeneralAnalysis.test_r()
#        GeneralAnalysis.blah_analysis()
#        GeneralAnalysis.determine_influential_variables()
        GeneralAnalysis.utm_object_analysis()
        
if __name__ == '__main__':
#    MRAnalysis.run()
    GeneralAnalysis.run()
    
