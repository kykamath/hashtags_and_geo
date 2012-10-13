'''
Created on Sep 26, 2012

@author: krishnakamath
'''
from datetime import datetime
from dateutil.relativedelta import relativedelta
from library.file_io import FileIO
from library.graphs import clusterUsingMCLClustering
from library.mrjobwrapper import runMRJob
from mr_analysis import ChiSquareTest
from mr_analysis import DemoAssociatioMeasure
from mr_analysis import FisherExactTest
from mr_analysis import HashtagsExtractor
from mr_analysis import PARAMS_DICT
from mr_analysis import WordObjectExtractor
from mr_analysis import WordHashtagContingencyTableObjectExtractor
from operator import itemgetter
from pprint import pprint
from settings import f_chi_square_association_measure
from settings import f_demo_association_measure
from settings import f_fisher_exact_association_measure
from settings import f_hashtags_extractor
from settings import f_hdfs_hashtags
from settings import f_word_objects_extractor
from settings import f_word_hashtag_contigency_table_objects
from settings import hdfs_input_folder
import networkx as nx
import time

class MRAnalysis():
    @staticmethod
    def run_job(mr_class, output_file, input_files_start_time, input_files_end_time):
        PARAMS_DICT['input_files_start_time'] = time.mktime(input_files_start_time.timetuple())
        PARAMS_DICT['input_files_end_time'] = time.mktime(input_files_end_time.timetuple())
        print 'Running map reduce with the following params:', pprint(PARAMS_DICT)
        runMRJob(
                     mr_class,
                     output_file,
                     MRAnalysis.get_input_files_with_tweets(input_files_start_time, input_files_end_time),
                     jobconf={'mapred.reduce.tasks':500}
                 )
        FileIO.writeToFileAsJson(PARAMS_DICT, output_file)
    @staticmethod
    def run_job_on_hashtags_in_dfs(mr_class, output_file):
        job_conf={'mapred.reduce.tasks':500, 'mapred.task.timeout': 86400000}
        print 'Running map reduce with the following params:'
        pprint(PARAMS_DICT)
        print 'Hadoop job conf:'
        pprint(job_conf)
        runMRJob(mr_class, output_file, [f_hdfs_hashtags], jobconf=job_conf)
        FileIO.writeToFileAsJson(PARAMS_DICT, output_file)
    @staticmethod
    def get_input_files_with_tweets(startTime, endTime, folderType='world'):
        current=startTime
        while current<=endTime:
            input_file = hdfs_input_folder%folderType+'%s_%s'%(current.year, current.month)
            print input_file
            yield input_file
            current+=relativedelta(months=1)
    @staticmethod
    def hashtags_extractor(input_files_start_time, input_files_end_time):
        mr_class = HashtagsExtractor
        output_file = f_hashtags_extractor
        MRAnalysis.run_job(mr_class, output_file, input_files_start_time, input_files_end_time)
    @staticmethod
    def word_object_extractor():
        mr_class = WordObjectExtractor
        output_file = f_word_objects_extractor
        runMRJob(
                     mr_class,
                     output_file,
                     [f_hdfs_hashtags],
                     jobconf={'mapred.reduce.tasks':500, 'mapred.task.timeout': 86400000}
                 )
    @staticmethod
    def word_object_contingency_table_extractor():
        mr_class = WordHashtagContingencyTableObjectExtractor
        output_file = f_word_hashtag_contigency_table_objects
        runMRJob(
                     mr_class,
                     output_file,
                     [f_hdfs_hashtags],
                     jobconf={'mapred.reduce.tasks':500, 'mapred.task.timeout': 86400000}
                 )
    @staticmethod
    def demo_association_measure():
        MRAnalysis.run_job_on_hashtags_in_dfs(DemoAssociatioMeasure, f_demo_association_measure)
    @staticmethod
    def fisher_exact_association_measure():
        MRAnalysis.run_job_on_hashtags_in_dfs(FisherExactTest, f_fisher_exact_association_measure)
    @staticmethod
    def chi_square_association_measure():
        MRAnalysis.run_job_on_hashtags_in_dfs(ChiSquareTest, f_chi_square_association_measure)
    @staticmethod
    def run():
        input_files_start_time, input_files_end_time = \
                                datetime(2011, 2, 1), datetime(2012, 8, 31)
#        MRAnalysis.hashtags_extractor(input_files_start_time, input_files_end_time)
#        MRAnalysis.word_object_extractor()
#        MRAnalysis.word_object_contingency_table_extractor()
#        MRAnalysis.demo_association_measure()
        MRAnalysis.fisher_exact_association_measure()
#        MRAnalysis.chi_square_association_measure()

def start_end(hashtag):
    return hashtag[:6], hashtag[-7:]

def get_components(graph):
    if graph.number_of_nodes()>5:
        try:
            for cluster in clusterUsingMCLClustering(graph): yield cluster
        except: yield graph.nodes()
    else: yield graph.nodes()

class HashtagGroupAnalysis(object):
    @staticmethod
    def temp():
        for line_no, data in\
                enumerate(FileIO.iterateJsonFromFile(f_chi_square_association_measure, remove_params_dict=True)):
            _, _, edges = data
            graph = nx.Graph()
#            try:
            for edge in edges: 
                u,v,attr_dict = edge
#                u, v = unicode(u).encode('utf-8'), unicode(v).encode('utf-8')
#                if start_end(u)!=('#promo', 'rubinho') and start_end(v)!=('#promo', 'rubinho') : 
#                    print unicode(u).encode('utf-8'), u
#                    print unicode(v).encode('utf-8'), v
    #                a = ''
    #                a.encode
                u = unicode(u).encode('utf-8')
                v = unicode(v).encode('utf-8')
#                print u, v
                graph.add_edge(u,v, attr_dict)
#            except: pass
#            ltuo_node_id_and_degree = graph.degree().items()
#            ltuo_node_id_and_degree.sort(key=itemgetter(1), reverse=True)
            print list(get_components(graph))
#            try:
#                print graph.number_of_nodes()
#                clusters = clusterUsingMCLClustering(graph)
#                print line_no, len(clusters)
#            except: print 'Exception'
#            ltuo_cluster_len_and_cluster = [(len(c), sorted(c)) for c in clusters]
#            ltuo_cluster_len_and_cluster.sort(key=itemgetter(0), reverse=True)
#            print zip(*ltuo_cluster_len_and_cluster)[0]
#            nx.write_dot(graph, 'graph.dot')
#            print len(edges), graph.number_of_edges()
#            exit()
    @staticmethod
    def run():
        HashtagGroupAnalysis.temp()

if __name__ == '__main__':
    MRAnalysis.run()
#    HashtagGroupAnalysis.run()
