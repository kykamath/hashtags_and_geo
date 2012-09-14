from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
import rpy2.rlike.container as rlc
import rpy2.robjects as robjects
import time


from library.file_io import FileIO
R = robjects.r

state = robjects.DataFrame.from_csvfile('/Users/kykamath/temp/state.df')
print state
#state_t = robjects.DataFrame(R.t(state))

g = R.lm('Life.Exp ~ Population + Income + Illiteracy + Murder + HS.Grad + Frost + Area', data=state)
summary = R.summary(g)
print summary.rx2('coefficients')

g = R.lm('Life.Exp ~ Population + Income + Illiteracy + Murder + HS.Grad + Frost', data=state)
summary = R.summary(g)
print summary.rx2('coefficients')


#stats = importr('stats')
#base = importr('base')
#faraway = importr('faraway')


#statedata <- data.frame(state.x77,row.names=state.abb,check.names=T)

#pima = faraway.pima
#state = faraway.state

#print state

#robjects.globalenv["df"] = df

#gfit = R.lm('Species ~ Area + Elevation + Nearest + Scruz + Adjacent', gala)

#gfit = R.lm('Species ~ Area + Elevation + Nearest + Scruz + Adjacent', gala)
#val = R.summary(gfit)
##print val
#print val.colnames

#grdevices = importr('grDevices')
#
#grdevices.png(file="path/to/file.png", width=512, height=512)
## plotting code here
#grdevices.dev_off()


#ctl = FloatVector([4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14])
#trt = FloatVector([4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69])
#group = R.gl(2, 10, 20, labels = ["Ctl","Trt"])
#weight = ctl + trt

#od = rlc.OrdDict([('group', group), ('weight', weight)])
#df = robjects.DataFrame(od)
#df.to_csvfile('abc.df')

#print robjects.DataFrame.from_csvfile('abc.df')

#FileIO.writeToFile(str(df), 'abc.df')

#data = R.read.table('abc.df')

#robjects.DataFrame.toread_data('abc,df')
#print data

#print str(df)

#print pima

#print robjects.r.summary(pima)

#R.hist(pima.rx2('diastolic'))


# LINEAR REGRSSION
#ctl = FloatVector([4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14])
#trt = FloatVector([4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69])
#group = R.gl(2, 10, 20, labels = ["Ctl","Trt"])
#weight = ctl + trt
#
#od = rlc.OrdDict([('group', group), ('weight', weight)])
#df = robjects.DataFrame(od)
#
#robjects.globalenv["df"] = df
#robjects.globalenv["weight"] = weight
#robjects.globalenv["group"] = group
#lm_D9 = R.lm("weight ~ group")
#lm_D91 = R.lm("weight ~ .", df)
#print(R.anova(lm_D9))
#print(R.anova(lm_D91))

## omitting the intercept
#lm_D90 = stats.lm("weight ~ group - 1")
#print(base.summary(lm_D90))

