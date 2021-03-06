'''
Created on Sept 9, 2012

@author: kykamath
'''
from collections import defaultdict
from datetime import datetime
from dateutil.relativedelta import relativedelta
from library.classes import GeneralMethods
from library.file_io import FileIO
from library.mrjobwrapper import runMRJob
from library.mrjobwrapper import runMRJobAndYieldResult
from library.r_helper import R_Helper
from mr_analysis import HashtagsByUTMId
from mr_analysis import HashtagsDistributionInUTM
from mr_analysis import HashtagsExtractor
from mr_analysis import HastagsWithUTMIdObject
from mr_analysis import PARAMS_DICT
from mr_analysis import SignificantNeirghborUTMIds
from mr_analysis import TweetStats
from operator import itemgetter
from pprint import pprint
from settings import f_hashtags_by_utm_id
from settings import f_hashtag_dist_by_accuracy
from settings import f_hashtags_extractor
from settings import f_hashtags_with_utm_id_object
from settings import f_significant_nei_utm_ids
from settings import f_tweet_stats
from settings import fld_google_drive_data_analysis
from settings import hdfs_input_folder
import cjson
import numpy as np
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
    def significant_nei_utm_ids():
        input_file = hdfs_input_folder%'generate_data_for_significant_nei_utm_ids'+\
                                                                'generate_data_for_significant_nei_utm_ids.json'
        print input_file
        print f_significant_nei_utm_ids
        runMRJob(SignificantNeirghborUTMIds,
                 f_significant_nei_utm_ids,
                 [input_file],
                 jobconf={'mapred.reduce.tasks':50, 'mapred.task.timeout': 86400000})
    
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
#        MRAnalysis.hashtags_by_utm_id(input_files_start_time,
#                                      input_files_end_time)
#        MRAnalysis.hashtags_with_utm_id_object(input_files_start_time,
#                                               input_files_end_time)
        MRAnalysis.significant_nei_utm_ids()
    
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
        ltuo_utm_id_and_num_of_neighbors_and_mean_common_h_count = []
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()+'.df'
        so_valid_utm_ids = set()
        for utm_object in FileIO.iterateJsonFromFile(f_hashtags_by_utm_id, True): 
            so_valid_utm_ids.add(utm_object['utm_id'])
        for utm_object in FileIO.iterateJsonFromFile(f_hashtags_by_utm_id, True):
            so_valid_nei_utm_ids = set(utm_object['mf_nei_utm_id_to_common_h_count']).intersection(so_valid_utm_ids)
            mean_num_of_common_h_count = np.mean([utm_object['mf_nei_utm_id_to_common_h_count'][nei_utm_id] 
                                               for nei_utm_id in so_valid_nei_utm_ids])
            ltuo_utm_id_and_num_of_neighbors_and_mean_common_h_count.append([utm_object['utm_id'], 
                                                                             len(so_valid_nei_utm_ids),
                                                                             mean_num_of_common_h_count])
        utm_ids, num_of_neighbors, mean_common_h_count = zip(*ltuo_utm_id_and_num_of_neighbors_and_mean_common_h_count)
        od = rlc.OrdDict([
                          ('utm_ids', robjects.StrVector(utm_ids)),
                          ('num_of_neighbors', robjects.FloatVector(num_of_neighbors)),
                          ('mean_common_h_count', robjects.FloatVector(mean_common_h_count))
                        ])
        df = robjects.DataFrame(od)
        FileIO.createDirectoryForFile(output_file)
        print 'Saving df to: ', output_file
        df.to_csvfile(output_file)
#    @staticmethod
#    def determine_influential_variables():
#        x = robjects.FloatVector([random.random() for i in range(10)])
#        y = robjects.FloatVector([random.random() for i in range(10)])
##        x1 =  y + robjects.r.rnorm(10)
#        od = rlc.OrdDict([('x', x), ('y', y)])
#        df = robjects.DataFrame(od)
#        df.rownames = 'a b c d e f g h i j'.split()
##        print x.r_repr()
##        print x.r_repr()
##        print len(df), df[0]
##        print df.r_repr()
#        print df.nrow
#        print df
#        
##        robjects.r.png('file.png')
##        robjects.r.plot(x,y)
##        robjects.r['dev.off']()
##        print df.rx(robjects.IntVector(range(1,100)), 'x')
    @staticmethod
    def significant_nei_utm_ids():
        mf_utm_id_to_valid_nei_utm_ids = {}
        def get_utm_vectors():
            so_hashtags = set()
            for utm_object in \
                    FileIO.iterateJsonFromFile(f_hashtags_by_utm_id, True):
                for hashtag, count in utm_object['mf_hashtag_to_count'].iteritems():
                    if hashtag!='total_num_of_occurrences': so_hashtags.add(hashtag)
                mf_utm_id_to_valid_nei_utm_ids[utm_object['utm_id']] =\
                                                                utm_object['mf_nei_utm_id_to_common_h_count'].keys()
            hashtags, ltuo_utm_id_and_vector = sorted(list(so_hashtags)), []
            for i, utm_object in enumerate(FileIO.iterateJsonFromFile(f_hashtags_by_utm_id, True)):
#                print i, utm_object['utm_id']
                utm_id_vector =  map(lambda hashtag: utm_object['mf_hashtag_to_count'].get(hashtag, 0.0),
                                     hashtags)
                ltuo_utm_id_and_vector.append((utm_object['utm_id'], 
                                               robjects.FloatVector(utm_id_vector)))
            od = rlc.OrdDict(sorted(ltuo_utm_id_and_vector, key=itemgetter(0)))
            df_utm_vectors = robjects.DataFrame(od)
            return df_utm_vectors
        output_file = fld_google_drive_data_analysis%GeneralMethods.get_method_id()
        df_utm_vectors = get_utm_vectors()
#        print df_utm_vectors.nrow
#        exit()
        utm_colnames = df_utm_vectors.colnames
        mf_utm_id_to_utm_colnames = dict(zip(sorted(mf_utm_id_to_valid_nei_utm_ids), utm_colnames))
        mf_utm_colnames_to_utm_id = dict(zip(utm_colnames, sorted(mf_utm_id_to_valid_nei_utm_ids)))
        for i, utm_colname in enumerate(utm_colnames):
            utm_id = mf_utm_colnames_to_utm_id[utm_colname]
            prediction_variable = utm_colname
            print i, utm_id
            predictor_variables = [mf_utm_id_to_utm_colnames[valid_nei_utm_ids]
                                    for valid_nei_utm_ids in mf_utm_id_to_valid_nei_utm_ids[utm_id]
                                        if valid_nei_utm_ids in mf_utm_id_to_utm_colnames and
                                           valid_nei_utm_ids != utm_id ]
            selected_utm_colnames =  R_Helper.variable_selection_using_backward_elimination(
                                                                                               df_utm_vectors,
                                                                                               prediction_variable,
                                                                                               predictor_variables,
                                                                                               debug=True
                                                                                            )
            nei_utm_ids = [mf_utm_colnames_to_utm_id[selected_utm_colname]
                                for selected_utm_colname in selected_utm_colnames]
            print 'Writing to: ', output_file
            FileIO.writeToFileAsJson({'utm_id': utm_id, 'nei_utm_ids': nei_utm_ids}, output_file)
    @staticmethod
    def generate_data_for_significant_nei_utm_ids():
        output_file = GeneralMethods.get_method_id()+'.json'
        so_hashtags, mf_utm_id_to_valid_nei_utm_ids = set(), {}
        for utm_object in \
                FileIO.iterateJsonFromFile(f_hashtags_by_utm_id, True):
            for hashtag, count in utm_object['mf_hashtag_to_count'].iteritems():
                if hashtag!='total_num_of_occurrences': so_hashtags.add(hashtag)
            mf_utm_id_to_valid_nei_utm_ids[utm_object['utm_id']] =\
                                                            utm_object['mf_nei_utm_id_to_common_h_count'].keys()
        hashtags = sorted(list(so_hashtags))
        mf_utm_id_to_vector = {}
        for utm_object in FileIO.iterateJsonFromFile(f_hashtags_by_utm_id, True):
#                print i, utm_object['utm_id']
            utm_id_vector =  map(lambda hashtag: utm_object['mf_hashtag_to_count'].get(hashtag, 0.0),
                                 hashtags)
            mf_utm_id_to_vector[utm_object['utm_id']] = robjects.FloatVector(utm_id_vector)
        for i, (utm_id, vector) in enumerate(mf_utm_id_to_vector.iteritems()):
            print '%s of %s'%(i+1, len(mf_utm_id_to_vector))
            ltuo_utm_id_and_vector = [(utm_id, vector)]
            for valid_nei_utm_id in mf_utm_id_to_valid_nei_utm_ids[utm_id]:
                if valid_nei_utm_id in mf_utm_id_to_vector and valid_nei_utm_id!=utm_id:
                    ltuo_utm_id_and_vector.append((valid_nei_utm_id, mf_utm_id_to_vector[valid_nei_utm_id]))
            od = rlc.OrdDict(sorted(ltuo_utm_id_and_vector, key=itemgetter(0)))
            df_utm_vectors = robjects.DataFrame(od)
            df_utm_vectors_json = R_Helper.get_json_for_data_frame(df_utm_vectors)
            dfm_dict = cjson.decode(df_utm_vectors_json)
            mf_utm_ids_to_utm_colnames = dict(zip(zip(*ltuo_utm_id_and_vector)[0], df_utm_vectors.colnames))
            utm_id_colname = mf_utm_ids_to_utm_colnames[utm_id]
            dfm_dict['prediction_variable'] = utm_id_colname
            dfm_dict['predictor_variables'] = filter(lambda colname: colname!=utm_id_colname,
                                                     df_utm_vectors.colnames)
            dfm_dict['mf_utm_colnames_to_utm_ids'] = dict(zip(df_utm_vectors.colnames, zip(*ltuo_utm_id_and_vector)[0]))
            FileIO.writeToFileAsJson(dfm_dict, output_file)
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
        GeneralAnalysis.significant_nei_utm_ids()
#        GeneralAnalysis.generate_data_for_significant_nei_utm_ids()
#        GeneralAnalysis.determine_influential_variables()
#        GeneralAnalysis.utm_object_analysis()
        
if __name__ == '__main__':
    MRAnalysis.run()
#    GeneralAnalysis.run()
    
