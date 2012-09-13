'''
Created on Sep 13, 2012

@author: krishnakamath
'''
from rpy2 import robjects
from rpy2.robjects import Formula
from rpy2.robjects.vectors import IntVector, FloatVector
from rpy2.robjects.lib import grid
from rpy2.robjects.packages import importr

import time

# The R 'print' function
#rprint = robjects.globalenv.get("print")
#stats = importr('stats')
#grdevices = importr('grDevices')
#base = importr('base')
#lattice = importr('lattice')
#
#xyplot = lattice.xyplot
#
#datasets = importr('datasets')
#mtcars = datasets.mtcars
#formula = Formula('mpg ~ wt')
#formula.getenvironment()['mpg'] = mtcars.rx2('mpg')
#formula.getenvironment()['wt'] = mtcars.rx2('wt')
#
#p = lattice.xyplot(formula)
#rprint(p)

R = robjects.r

print R.rnorm(1000)

time.sleep(100)