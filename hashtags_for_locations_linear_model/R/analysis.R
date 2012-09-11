# Created on Sept 11, 2012
# Author: Krishna Y. Kamath

# Function to get json object.
# Example:
# json_file <- "http://webonastick.com/uscl/feeds/uscl.json.txt"
# getJSONObject(json_file)
getJSONObject <- function(json_file) {
	library("rjson")
	return(fromJSON(paste(readLines(json_file), collapse="")))
}

json_file <- "http://webonastick.com/uscl/feeds/uscl.json.txt"
getJSONObject(json_file)