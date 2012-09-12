# Created on Sept 11, 2012
# Author: Krishna Y. Kamath

source('../../R/library.R')

getJSONObject <- function(json_file) {
	library("rjson")
	print(paste("Reading json from", json_file, sep = " "))
	return(fromJSON(paste(readLines(json_file), collapse="")))
}

#data <- getJSONObject('/Users/kykamath/Documents/workspace_sept_2012/hashtags_and_geo/R/data.json')
data <- getJSONObject('/Users/kykamath/Documents/workspace_sept_2012/hashtags_and_geo/hashtags_for_locations_linear_model/hashtags_with_utm_id_object')
data
#data$menuitem[-length(data$menuitem)]
#getDataFrameFromListOfDicts(data$menuitem)