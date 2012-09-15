# Created on Sept 11, 2012
# Author: Krishna Y. Kamath

#source('../../R/library.R')

utm_object_analysis <- function() {
	utm_object_analysis = read.csv(paste('~/Google Drive/Desktop/hashtags_and_geo/hashtags_for_locations_linear_model/',
										'GeneralAnalysis/utm_object_analysis.df', sep=''))
	hist(utm_object_analysis$num_of_neighbors)
}

utm_object_analysis()