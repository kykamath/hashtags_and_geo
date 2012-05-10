##for spammerPercentage in range(1,21):
##        spammerPercentage = spammerPercentage* 0.05
##        print spammerPercentage
#
#
#l1 = [spammerPercentage* 0.001 for spammerPercentage in range(1,51)]
#l2 = [spammerPercentage* 0.05 for spammerPercentage in range(1,21)]

#
#import random
#import matplotlib.pyplot as plt
#
#x0 = 0. 
#x1 = 1.
#n=3.01
#l = []
#for i in range(5000):
#    y = random.random()
#    l.append(1- ((x1**(n+1) - x0**(n+1))*y + x0**(n+1))**(1/(n+1)))
#    
#plt.hist(l,100)
##plt.loglog(0,l[1])
#plt.xlim(xmax=1)
#plt.show()


#from pylab import *
#
#gaussian = lambda x: 3*exp(-(30-x)**2/20.)
#
#data = gaussian(arange(100))
#
#plot(data)
#
#X = arange(data.size)
#x = sum(X*data)/sum(data)
#width = sqrt(abs(sum((X-x)**2*data)/sum(data)))
#
#max = data.max()
#
#fit = lambda t : max*exp(-(t-x)**2/(2*width**2))
#
#plot(fit(X))
#
#show()


import random
import matplotlib.pyplot as plt
#
l = []
val=-1
for i in range(1000):
#    while val<0.5:
#    val = random.gauss(0.5, 0.01)
    val = random.paretovariate(0.76)
#    if val>0.5:
#    print val
    l.append(val)
print l
plt.hist(l,100)
#plt.xlim(xmax=1.0)
plt.show()