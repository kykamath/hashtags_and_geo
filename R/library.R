# Created on Sept 11, 2012
# Author: Krishna Y. Kamath
# 
# A collection of useful R functions
#

# Function to get json object.
# Example:
# json_file <- "http://webonastick.com/uscl/feeds/uscl.json.txt"
# getJSONObject(json_file)
getJSONObject <- function(json_file) {
	library("rjson")
	return(fromJSON(paste(readLines(json_file), collapse="")))
}

#Given a list of dictionary like objects, this method coverts it to a data
#frame.
#Example:
#		Given:
#       > data <- getJSONObject("/Users/krishnakamath/Documents/
#							workspace_sept_12/hashtags_and_geo/R/data.json")
#		> data 
#			$menuitem
#			$menuitem[[1]]
#			$menuitem[[1]]$value
#			[1] "New"
#			
#			$menuitem[[1]]$onclick
#			[1] "CreateNewDoc()"
#			
#			
#			$menuitem[[2]]
#			$menuitem[[2]]$value
#			[1] "Open"
#			
#			$menuitem[[2]]$onclick
#			[1] "OpenDoc()"
#			
#			
#			$menuitem[[3]]
#			$menuitem[[3]]$value
#			[1] "Close"
#			
#			$menuitem[[3]]$onclick
#			[1] "CloseDoc()"
#
#		> getDataFrameFromListOfDicts(data$menuitem)
#		value        onclick
#		1   New CreateNewDoc()
#		2  Open      OpenDoc()
#		3 Close     CloseDoc()
getDataFrameFromListOfDicts <- function(list_of_dicts) {
	return(data.frame(t(sapply(list_of_dicts, unlist))))
}
