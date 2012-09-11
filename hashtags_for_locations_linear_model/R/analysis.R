# Created on Sept 11, 2012
# Author: Krishna Y. Kamath

source('../../R/library.R')

data <- getJSONObject(paste('/Users/krishnakamath/Documents/workspace_sept_12/',
					  		'hashtags_and_geo/R/data.json', sep=''))
data$menuitem[-length(data$menuitem)]
#getDataFrameFromListOfDicts(data$menuitem)